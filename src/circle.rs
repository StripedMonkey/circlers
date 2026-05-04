use std::{
    ffi::{CStr, CString},
    os::{fd::AsFd, unix::ffi::OsStringExt},
    path::{Path, PathBuf},
    sync::Arc,
};

use ferrompi::Communicator;
use futures::{FutureExt as _, select};
use nix::dir::Entry;
use rand::{Rng as _, seq::SliceRandom as _};
use smol::pin;
use tracing::{debug, error, trace};

use crate::{
    CircleError, WorkResponse,
    async_mpi::{self, SOURCE_ANY, Tag},
    termination_detector::TerminationDetectionState,
    walker,
};

// The main loop of the walk consists of the following tasks, running asynchronously. Note not all tasks are run all the
// time, as they're simply not necessary.
// 1. Walking the directory tree, processing entries as they are encountered, a directory at a time. The walk
//    is directory-wide, depth first, and distributed across the ranks. To perform the minimum number of
//    syscalls possible, we iterate over the directory entries, stashing the directories for later processing.
//    after processing the entries in the current directory, we stash all but one of the directories in a queue
//    for later processing or distribution to other ranks, and then we process the remaining directory. If there
//    are no directories, we pop a directory from the queue and work on that one.
//    For each entry, we call a user-provided callback as appropriate
//
// 2. Responding to messages from other ranks, which consists of
//    a. Requesting work (directories to process) from other ranks, if we are idle and have no work in our queue
//    b. Responding to work requests from other ranks, if we have work in our queue, we split off a portion of
//       our queue and send it to the requesting rank. The exact portion of the queue to send is something I
//       would like to experiment with. According to
//       [When Random is Better: Parallel File Tree Walking](http://jlafon.io/parallel-file-treewalk-part-II.html)
//       randomized queue splitting performs better than splitting in half every time. This makes some sense,
//       but surely there are even better strategies. It seems like the author tried the random queue splitting
//       and that was fast enough that the syscall overhead dominated, so they didn't continue further.
//
// 3. "Making progress" on termination detection. The program is considered terminated when
//    a. All processes are "locally terminated"
//    b. There are no messages in transit

pub struct Circle {
    comm: Communicator,
    termination_state: TerminationDetectionState,
}

impl Circle {
    pub fn new(world: &Communicator) -> Result<Self, ferrompi::Error> {
        let comm = world.duplicate()?;
        let term_detector = TerminationDetectionState::new(&comm);
        Ok(Circle {
            comm,
            termination_state: term_detector,
        })
    }

    // Walk the provided file tree starting at `path`. When no path is provided, request work from other ranks until
    // termination is signaled. This function does not return until termination is detected and all participating ranks
    // exit gracefully.
    pub async fn start_walk<F1, F2>(
        &mut self,
        path: Option<&Path>,
        on_file_entry: F1,
        on_dir_entry: F2,
    ) -> Result<(), CircleError>
    where
        F1: FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
        F2: FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
    {
        let mut walker = walker::Walker::new(on_file_entry, on_dir_entry);

        if let Some(path) = path {
            walker.extend_queue([path]).await.unwrap();
        }

        // At this point, we have two tasks we need to run concurrently:
        // - The directory walk
        // - Work request handling
        // We don't need to handle termination detection since we know termination can't occur until we're locally idle.
        // any termination progress can be made after we're idle.
        let handle_work_requests = self.work_loop(walker.get_queue()).fuse();
        pin!(handle_work_requests);
        loop {
            {
                let walker_worker = walker.work_directory_queue().fuse();
                pin!(walker_worker);
                select! {
                    res = walker_worker => {
                        if let Err(e) = res {
                            error!("Error walking directory queue: {:?}", e);
                            // We only propagate critical errors here
                            panic!("{e:?}"); // TODO: bubble errors up if it makes sense to.
                        }
                    },
                    e = handle_work_requests => {
                        panic!("Work request handling should never complete: {:?}", e);
                    },
                }
            }
            debug_assert!(
                walker.queue_len() == 0,
                "
                After completing a walk task, the queue should be empty since we should have stashed all directories for
                later processing or distribution.
                "
            );
            // We've completed all the work we have available to us, and we have to wait for more work to arrive or for
            // termination to be detected.
            // We need to now start making progress on termination detection
            if self.comm.size() == 1 {
                debug!("Only one rank in communicator, skipping termination detection");
                break Ok(());
            }
            debug!("Locally idle, polling for more work or termination");
            // TODO: We initialize the termination detection task here, since we don't need to make progress on
            // termination until we're locally idle, but this means that we drop our termination_detection task each
            // loop. Validate whether this can cause issues if we drop the task at the wrong time. Initial testing seems
            // to indicate this doesn't cause any permanent hangs, but it could simply be rare.
            let handle_termination_detection = self.detect_termination().fuse();
            pin!(handle_termination_detection);
            select! {
                // Even though we have no work to give out, we still need to reject work requests from other ranks to
                // prevent deadlocks/cyclic work requests.
                e = handle_work_requests => {
                    panic!("Work request handling task should never complete: {:?}", e);
                },
                // If we receive work, we're no longer idle, and we can go back to work by looping.
                work = self.request_work().fuse() => match work {
                    Ok(work) => {
                        debug!("Received {} work items from another rank", work.len());
                        walker.extend_queue(work).await.unwrap();
                        continue;
                    },
                    Err(e) => {
                        panic!("Error requesting work from other ranks: {}", e);
                    }
                },
                // We made enough progress on termination that it's been detected, exit gracefully.
                t = handle_termination_detection => {
                    if let Err(e) = t {
                        panic!("Error on detecting termination: {:?}", e);
                    }
                    // Termination was detected, and we're currently idle, so we can terminate gracefully.
                    debug!("Termination detected, exiting");
                    break Ok(());
                },
            }
        }
    }

    /// Request work from other ranks. This will retry until it receives work from another rank.
    async fn request_work(&self) -> Result<Vec<PathBuf>, CircleError> {
        let rank = self.comm.rank();
        let size = self.comm.size();

        loop {
            // TODO: Make it easier to choose the work request strategy for benchmarking purposes.
            // The current method chooses a random order each loop to request work from, theoretically ensuring that all
            // ranks have an equal probability of choosing each other rank.
            // Choose a random priority order to request work from.
            let mut candidates: Vec<i32> =
                (0..size).filter(|&candidate| candidate != rank).collect();
            let mut rng = rand::rng();
            candidates.shuffle(&mut rng);

            for candidate in candidates {
                let request_payload = [0u8];
                async_mpi::isend(
                    &self.comm,
                    &request_payload,
                    candidate,
                    Tag::WorkRequest as i32,
                )
                .await?;

                let (_source, response_buf) =
                    async_mpi::receive_tagged(&self.comm, candidate, Tag::WorkResponse as i32)
                        .await?;

                let response: WorkResponse = postcard::from_bytes(&response_buf)?;
                if let WorkResponse::Work(paths) = response {
                    let work = paths
                        .into_iter()
                        .map(|path| PathBuf::from(std::ffi::OsString::from_vec(path)))
                        .collect();

                    // Ack that we received work from `candidate`.
                    let ack_payload = [0u8];
                    async_mpi::isend(&self.comm, &ack_payload, candidate, Tag::WorkAck as i32)
                        .await?;

                    return Ok(work);
                }
            }
            debug!("Looping through all candidates for work didn't yield any work, retrying...");
        }
    }

    /// Handle incoming work requests and acks from ranks we've given work to.
    async fn work_loop(
        &self,
        queue: Arc<smol::lock::Mutex<Vec<CString>>>,
    ) -> Result<(), CircleError> {
        // Continuously handle four types of incoming messages:
        // 1. WorkRequest: respond with work from our queue or reject
        // 2. WorkAck: receive acknowledgment that our work was received
        // 3. Token: receive the token for distributed termination detection
        // 4. TerminationConfirmed: receive confirmation that termination has been detected and we can exit
        loop {
            select! {
                work_request_status = async_mpi::receive_tagged(&self.comm, SOURCE_ANY, Tag::WorkRequest as i32).fuse() => {
                    let (source, _work_request_status): (i32, Vec<u8>) = work_request_status?;
                    let response = {
                        let mut queue_guard = queue.lock().await;
                        if queue_guard.is_empty() {
                            trace!("Rejecting WorkRequest from rank {} since we have no work to share", source);
                            WorkResponse::Reject
                        } else {
                            // If we're sending work, mark ourselves as black and increment pending receipts
                            self.termination_state.on_message_sent().await;
                            let mut rng = rand::rng();
                            // Keep at least one item in the queue for ourselves, and share the rest.
                            let share_count = rng.random_range(1..=queue_guard.len());
                            let mut shared_work = Vec::with_capacity(share_count);
                            for _ in 0..share_count {
                                let idx = rng.random_range(0..queue_guard.len());
                                shared_work.push(queue_guard.swap_remove(idx).into_bytes());
                            }
                            trace!("Sharing {} work items with rank {}", shared_work.len(), source);
                            WorkResponse::Work(shared_work)
                        }
                    };
                    let response_buf = postcard::to_allocvec(&response)?;
                    async_mpi::isend(&self.comm, &response_buf, source, Tag::WorkResponse as i32).await?;
                },
                work_ack = async_mpi::receive_tagged(&self.comm, SOURCE_ANY, Tag::WorkAck as i32).fuse() => {
                    let (source, _work_ack_status): (i32, Vec<u8>) = work_ack?;
                    self.termination_state.on_receipt_received();
                    trace!(
                        "Received WorkAck from rank {}",
                        source,
                    );
                },
            };
            trace!("Loop!~")
        }
    }

    async fn detect_termination(&self) -> Result<(), CircleError> {
        self.termination_state.monitor_termination(&self.comm).await
    }
}

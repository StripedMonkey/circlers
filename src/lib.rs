pub mod async_mpi;
pub mod termination_detector;
pub mod walker;

use std::{
    ffi::CString,
    os::unix::ffi::OsStringExt,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

use ferrompi::Communicator;
use futures::{FutureExt as _, select};
use rand::{Rng as _, seq::SliceRandom as _};
use serde::{Deserialize, Serialize};
use smol::pin;
use tracing::{debug, trace};

use crate::{termination_detector::TerminationDetectionState, walker::OnEntryCallback};

pub struct Circle {
    comm: Communicator,
    locally_idle: AtomicBool,
    termination_state: TerminationDetectionState,
    on_file_entry: OnEntryCallback,
    on_dir_entry: OnEntryCallback,
}

// TODO: There are libraries that make this kind of thing saner. Use one.
/// Tag for sending and receiving messages between ranks.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub(crate) enum Tag {
    WorkRequest = 10,
    WorkResponse,
    WorkAck,
    TerminationToken,
    TerminationConfirmed,
}

impl From<i32> for Tag {
    fn from(value: i32) -> Self {
        match value {
            10 => Tag::WorkRequest,
            11 => Tag::WorkResponse,
            12 => Tag::WorkAck,
            13 => Tag::TerminationToken,
            14 => Tag::TerminationConfirmed,
            _ => panic!("Invalid tag value: {value}"),
        }
    }
}

pub const SOURCE_ANY: i32 = -1;

#[derive(Debug, Serialize, Deserialize)]
enum WorkResponse {
    Work(Vec<Vec<u8>>),
    Reject,
}

#[derive(Debug)]
pub enum CircleError {
    Mpi(ferrompi::Error),
    Serialization(postcard::Error),
    InvalidMessageCount(i64),
    NoPeerRanks,
}

impl std::fmt::Display for CircleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircleError::Mpi(err) => write!(f, "MPI error: {err}"),
            CircleError::Serialization(err) => write!(f, "Serialization error: {err}"),
            CircleError::InvalidMessageCount(count) => {
                write!(f, "Invalid message count from MPI status: {count}")
            }
            CircleError::NoPeerRanks => write!(f, "Cannot request work without peer ranks"),
        }
    }
}

impl std::error::Error for CircleError {}

impl From<ferrompi::Error> for CircleError {
    fn from(value: ferrompi::Error) -> Self {
        Self::Mpi(value)
    }
}

impl From<postcard::Error> for CircleError {
    fn from(value: postcard::Error) -> Self {
        Self::Serialization(value)
    }
}

// The main loop of the walk consists of the following tasks, running asynchronously:
// 1. Walking the directory tree, processing entries as they are encountered, a directory at a time. The walk
//    is directory-wide, depth first, and distributed across the ranks. To perform the minimum number of
//    syscalls possible, we iterate over the directory entries, stashing the directories for later processing.
//    after processing the entries in the current directory, we stash all but one of the directories in a queue
//    for later processing or distribution to other ranks, and then we process the remaining directory. If there
//    are no directories, we pop a directory from the queue and work on that one.
//    For each entry, we call a user-provided callback as appropriate
//
// 2. "Making progress" on termination detection. The program is considered terminated when
//    a. All processes are "locally terminated"
//    b. There are no messages in transit
//
// 3. Responding to messages from other ranks, which consists of
//    a. Requesting work (directories to process) from other ranks, if we are idle and have no work in our queue
//    b. Responding to work requests from other ranks, if we have work in our queue, we split off a portion of
//       our queue and send it to the requesting rank. The exact portion of the queue to send is something I
//       would like to experiment with. According to
//       [When Random is Better: Parallel File Tree Walking](http://jlafon.io/parallel-file-treewalk-part-II.html)
//       randomized queue splitting performs better than splitting in half every time. This makes some sense,
//       but surely there are even better strategies. It seems like the author tried the random queue splitting
//       and that was fast enough that the syscall overhead dominated, so they didn't continue further.

impl Circle {
    pub fn new(world: &Communicator) -> Result<Self, ferrompi::Error> {
        let comm = world.duplicate()?;
        let term_detector = TerminationDetectionState::new(&comm);
        Ok(Circle {
            comm,
            locally_idle: AtomicBool::new(false),
            termination_state: term_detector,
            on_file_entry: |_, _, _| Ok(()),
            on_dir_entry: |_, _, _| Ok(()),
        })
    }

    pub fn set_on_file_entry(&mut self, callback: OnEntryCallback) {
        self.on_file_entry = callback;
    }

    pub fn set_on_dir_entry(&mut self, callback: OnEntryCallback) {
        self.on_dir_entry = callback;
    }

    // Walk the provided file tree starting at `path`. When no path is provided, request work from other ranks until
    // termination is signaled.
    pub async fn start_walk(&mut self, path: Option<&Path>) {
        let mut walker = walker::Walker::new();

        walker.set_on_file_entry(self.on_file_entry);
        walker.set_on_directory_entry(self.on_dir_entry);

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
            let walker_task = walker.work_directory_queue().fuse();
            pin!(walker_task);
            select! {
                res = walker_task => {
                    if let Err(e) = res {
                        panic!("Error walking directory queue: {:?}", e);
                    }
                },
                e = handle_work_requests => {
                    panic!("Work request handling should never complete: {:?}", e);
                },
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
            trace!("Locally idle, polling for more work or termination");
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
                        self.locally_idle.store(false, std::sync::atomic::Ordering::SeqCst);
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
                    break;
                },
            }
        }
    }

    /// Request work from other ranks. This will retry until it receives work from another rank.
    async fn request_work(&self) -> Result<Vec<PathBuf>, CircleError> {
        let rank = self.comm.rank();
        let size = self.comm.size();

        if size <= 1 {
            return Err(CircleError::NoPeerRanks);
        }

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

                let response_buf =
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
            // Check for WorkRequest
            let probe_work_request =
                async_mpi::probe_tag(&self.comm, SOURCE_ANY, Tag::WorkRequest as i32).fuse();
            let probe_work_ack =
                async_mpi::probe_tag(&self.comm, SOURCE_ANY, Tag::WorkAck as i32).fuse();
            pin!(probe_work_request);
            pin!(probe_work_ack);
            select! {
                work_request_status = probe_work_request => {
                    let work_request_status = work_request_status?;
                    trace!("Received WorkRequest from rank {}", work_request_status.source);
                    let mut request_buf = vec![0u8; work_request_status.count as usize];
                    async_mpi::irecv(&self.comm, &mut request_buf, work_request_status.source, Tag::WorkRequest as i32).await?;
                    let response = {
                        let mut queue_guard = queue.lock().await;
                        if queue_guard.is_empty() {
                            trace!("Rejecting WorkRequest from rank {} since we have no work to share", work_request_status.source);
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
                            trace!("Sharing {} work items with rank {}", shared_work.len(), work_request_status.source);
                            WorkResponse::Work(shared_work)
                        }
                    };
                    let response_buf = postcard::to_allocvec(&response)?;
                    async_mpi::isend(&self.comm, &response_buf, work_request_status.source, Tag::WorkResponse as i32).await?;
                },
                work_ack_status = probe_work_ack => {
                    let work_ack_status = work_ack_status?;
                    let mut ack_buf = vec![0u8; work_ack_status.count as usize];
                    async_mpi::irecv(&self.comm, &mut ack_buf, work_ack_status.source, Tag::WorkAck as i32).await?;
                    self.termination_state.on_receipt_received();
                    trace!(
                        "Received WorkAck from rank {}",
                        work_ack_status.source,
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

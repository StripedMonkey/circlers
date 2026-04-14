use std::{
    collections::VecDeque,
    ffi::OsStr,
    fmt::Debug,
    io, mem,
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use clap::Parser as _;
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Destination as _, Source as _},
};
use nix::{
    NixPath,
    dir::{Dir, Entry, Type},
    fcntl::OFlag,
    sys::stat::Mode,
};
use rand::Rng;
use tracing::{error, trace};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt as _, util::SubscriberInitExt};

#[derive(clap::Parser)]
struct Args {
    #[clap(default_value_os_t = PathBuf::from("./"))]
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
struct WorkMessage<T> {
    paths: Vec<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
#[repr(i32)]
enum WorkRequest {
    RequestWork = 1,
    WorkOffer = 2,
}

struct Worker<T> {
    comm: SimpleCommunicator,
    queue: VecDeque<T>,
    /// Function called when visiting a directory entry.
    ///
    /// This function is also the thing responsible for enqueueing new work based on whatever
    /// heuristics the user needs. This also allows the user to define the type of the path, so
    /// long as it implements [`NixPath`](nix::NixPath) and is encodable/decodable with bincode.
    visit_element: fn(&mut Self, &T, Entry),
    /// Function called when an error is encountered trying to visit a directory/file path.
    ///
    /// This can happen in two situations:
    /// 1. When trying to open a directory to read its entries, mainly due to permission errors.
    /// 2. When reading entries from a directory that have become invalid since the directory was
    ///    opened.
    visit_error: fn(&mut Self, &T, nix::Error),
    /// The number of work receipts that are currently pending. When offering work, we use this to
    /// track how many offers have not been accepted yet for termination detection.
    pending_work_receipts: usize,
}

impl<T> Worker<T> {
    /// Enqueue a new element to be processed by the worker.
    pub fn enqueue(&mut self, element: T) {
        self.queue.push_back(element);
    }

    /// Split off a number of elements from the current work queue
    fn split_work(&mut self, num_elements: usize) -> VecDeque<T> {
        // TODO: We do this for two assumptions:
        // 1. The first elements in the queue are going to be higher up in the directory tree
        // 2. The directories higher up in the tree are more likely to have more elements, meaning
        //    that giving them to another worker will better balance the load.
        // These assumptions seem logical, but I don't actually know whether or not it's true in
        // reality. The original inspiration for this project didn't make any assumptions
        let mut split = self.queue.split_off(self.queue.len() - num_elements);
        mem::swap(&mut split, &mut self.queue);
        split
    }
}

impl<T> Worker<T>
where
    T: Encode + Decode<()> + Debug + NixPath,
{
    fn process_entry(&mut self, path: &T) {
        // TODO: Investigate the performance impact of using O_DIRECT here
        let mut dir = match Dir::open(path, OFlag::O_DIRECTORY | OFlag::O_RDONLY, Mode::empty()) {
            Ok(dir) => dir,
            Err(e) => {
                return (self.visit_error)(self, path, e);
            }
        };
        for entry in dir.iter() {
            match entry {
                Ok(entry) => match entry.file_name().to_bytes() {
                    b"." | b".." => continue,
                    _ => (self.visit_element)(self, path, entry),
                },
                Err(e) => {
                    panic!("Error reading entry on {:#?}: {:?}", path, e);
                }
            }
        }
    }

    /// Process any incoming work requests from other workers, giving them work, if possible. If no
    /// work requests were processed, returns None.
    fn process_work_requests(&mut self) -> Option<()> {
        let any_process = self.comm.any_process();
        let mut current_work_requests = vec![];
        loop {
            let Some((message, status)) =
                any_process.immediate_matched_probe_with_tag(WorkRequest::RequestWork as i32)
            else {
                break;
            };

            let source_rank = status.source_rank();
            trace!("Received work request from {}", source_rank);
            message.matched_receive_into::<[u8]>(&mut [0u8; 1]);
            current_work_requests.push(source_rank);
        }
        if current_work_requests.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        // Split a random amount of work off to give to the requesters
        let amount_to_give = rng.random_range(1..self.queue.len());
        let data_to_give = self.split_work(amount_to_give);
        let num_requesters = current_work_requests.len();
        let amount_per_requester = data_to_give.len() / num_requesters;
        let mut remainder = data_to_give.len() % num_requesters;
        let mut data_iter = data_to_give.into_iter();
        for requester in current_work_requests {
            let mut to_send = amount_per_requester;
            if remainder > 0 {
                to_send += 1;
                remainder -= 1;
            }
            let mut send_data = Vec::with_capacity(to_send);
            send_data.extend((&mut data_iter).take(to_send));

            trace!(
                "Sending {} work items to requester {}",
                send_data.len(),
                requester
            );
            let serialized = bincode::encode_to_vec(&send_data, bincode::config::standard())
                .expect("Failed to serialize work message");
            self.comm
                .process_at_rank(requester)
                .send_with_tag(&serialized, WorkRequest::WorkOffer as i32);
        }

        Some(())
    }

    fn request_work(&mut self) {
        let size = self.comm.size();
        if size <= 1 {
            trace!("Only one worker in communicator, not requesting work");
            return;
        }
        let mut rng = rand::rng();
        let target_rank = loop {
            let rank = rng.random_range(0..size);
            if rank != self.comm.rank() {
                break rank;
            }
        };
        trace!("Requesting work from {}", target_rank);
        let other_process = self.comm.process_at_rank(target_rank);
        other_process.send_with_tag::<[u8]>([0].as_slice(), WorkRequest::RequestWork as i32);
        trace!("Waiting for work offer from {}", target_rank);
        let (buf, _status) = other_process.receive_vec_with_tag(WorkRequest::WorkOffer as i32);
        trace!(
            "Received work offer from {} ({} bytes)",
            target_rank,
            buf.len()
        );
        let (message, _count): (Vec<T>, usize) =
            bincode::decode_from_slice(&buf, bincode::config::standard()).unwrap();
        self.queue.extend(message);
    }

    /// Begin processing work items in the queue
    pub fn work(&mut self) {
        loop {
            // NOTE: The current implementation has to be non-blocking
            // TODO: Async with `select!` would be super nice. Implementing async

            self.process_work_requests();

            if self.queue.is_empty() {
                self.request_work();
            }

            match self.queue.pop_back() {
                Some(element) => self.process_entry(&element),
                None => {
                    trace!("Worker emptied queue");
                    break;
                }
            }
        }
    }
}

fn visit_element_f(worker: &mut Worker<PathBuf>, path: &PathBuf, entry: Entry) {
    match entry.file_type() {
        Some(Type::Directory) => {
            let mut new_path = path.clone();
            // SAFETY: We know that the file name is a valid Path segment because it came directly
            // from a path from the filesystem.
            new_path
                .push(unsafe { OsStr::from_encoded_bytes_unchecked(entry.file_name().to_bytes()) });
            worker.enqueue(new_path);
        }
        _ => {
            println!(
                "{}: {}",
                worker.comm.rank(),
                path.join(unsafe {
                    OsStr::from_encoded_bytes_unchecked(entry.file_name().to_bytes())
                })
                .display()
            );
        }
    }
}

fn visit_error_f<T>(worker: &mut Worker<T>, path: &T, error: nix::Error)
where
    T: NixPath + Debug,
{
    error!("Error reading directory {:#?}: {:?}", path, error);
}

impl Worker<PathBuf> {
    fn seeded_worker(mpi: SimpleCommunicator, root: PathBuf) -> Self {
        let comm = mpi.duplicate();
        comm.set_name("CircleWorkerComm");

        let mut queue = VecDeque::new();
        queue.push_back(root);
        Worker {
            comm,
            queue,
            visit_element: visit_element_f,
            visit_error: visit_error_f,
            pending_work_receipts: todo!(),
        }
    }
    fn unseeded_worker(mpi: SimpleCommunicator) -> Self {
        let comm = mpi.duplicate();
        comm.set_name("CircleWorkerComm");

        let queue = VecDeque::new();
        Worker {
            comm,
            queue,
            visit_element: visit_element_f,
            visit_error: visit_error_f,
            pending_work_receipts: todo!(),
        }
    }
}

fn main() {
    let tracing = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing)
        .init();
    let args = Args::parse();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut worker = match world.rank() {
        0 => Worker::seeded_worker(world, args.root),
        _ => Worker::unseeded_worker(world),
    };
    worker.work();
}

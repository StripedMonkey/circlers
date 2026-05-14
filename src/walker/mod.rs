use std::{
    ffi::{CStr, CString},
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd},
        unix::ffi::OsStrExt,
    },
    path::Path,
    rc::Rc,
};

use nix::{
    dir::{Dir, Entry, Type},
    fcntl::{AtFlags, OFlag},
    sys::stat::{self, Mode, SFlag},
};
use smol::{future::yield_now, lock::Mutex};
use tracing::warn;

#[derive(Debug)]
pub enum WalkerError {
    Nix(nix::Error),
}

impl From<nix::Error> for WalkerError {
    fn from(e: nix::Error) -> Self {
        WalkerError::Nix(e)
    }
}

impl std::fmt::Display for WalkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalkerError::Nix(e) => write!(f, "nix error: {e}"),
        }
    }
}

impl std::error::Error for WalkerError {}

/// An open directory whose subdirectory names have been enumerated but not yet walked.
///
/// `fd` holds a boxed `OwningIter` (produced by `Dir::into_iter()`). There is no way to extract the `Dir` back from an
/// `OwningIter` once iteration begins, so we erase its type to `dyn AsRawFd` while preserving ownership and the
/// underlying fd for future `Dir::openat` calls on each pending subdir. (`OwningIter` implements `AsRawFd` but not
/// `AsFd`, so we use the raw fd form and construct `BorrowedFd` manually when needed.)
struct DirectoryEntries {
    fd: Box<dyn AsRawFd>,
    dir_path: CString,
    pending_subdirs: Vec<CString>,
}

struct Queue {
    // TODO: The hot queue holds an opened file descriptor for each directory parent, this might be a lot. At the moment
    // because we're essentially doing a depth-first walk, we will only have depth(tree) fds open at once, which is a
    // reasonable-ish assumption but might cause issues. Consider ensuring that we limit the number of "hot" entries and
    // keep an upper bound number of items.
    /// Directories with open fds; children are opened via `Dir::openat(fd, name)`.
    hot: Vec<DirectoryEntries>,
    /// Full paths with no cached fd; opened via `Dir::open(full_path)`. Populated by external work injection (MPI or the
    /// initial `walk()` seed).
    cold: Vec<CString>,
}

impl Queue {
    fn new() -> Self {
        Queue {
            hot: Vec::new(),
            cold: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.cold.is_empty()
    }

    fn len(&self) -> usize {
        self.cold.len()
            + self
                .hot
                .iter()
                .map(|de| de.pending_subdirs.len())
                .sum::<usize>()
    }

    /// Extract up to `count` items as full paths. Drains `cold` first, then constructs paths from `hot` entries. `hot`
    /// entries that become empty are dropped, closing their fd.
    fn split_work(&mut self, count: usize) -> Vec<CString> {
        let mut out = Vec::with_capacity(count.min(self.len()));
        while out.len() < count {
            if let Some(cold) = self.cold.pop() {
                out.push(cold);
                continue;
            }
            match self.hot.last_mut() {
                None => break,
                Some(de) => {
                    let name = de.pending_subdirs.pop().unwrap();
                    out.push(combine_paths(&de.dir_path, &name));
                    if de.pending_subdirs.is_empty() {
                        self.hot.pop();
                    }
                }
            }
        }
        out
    }
}

pub struct Walker<F1, F2> {
    queue: Rc<Mutex<Queue>>,
    on_file: F1,
    on_dir: F2,
}

impl<F1, F2> Walker<F1, F2>
where
    F1: FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
    F2: FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
{
    pub fn new(on_file: F1, on_dir: F2) -> Self {
        Self {
            queue: Rc::new(Mutex::new(Queue::new())),
            on_file,
            on_dir,
        }
    }

    /// Return a cloneable handle to the internal queue for concurrent work-sharing.
    pub fn work_queue(&self) -> WorkQueue {
        WorkQueue(self.queue.clone())
    }

    /// Walk the file tree rooted at `root` to completion. Non-fatal filesystem errors (permission denied, races, etc.)
    /// are logged and skipped; callback errors propagate.
    pub async fn walk(&mut self, root: &Path) -> Result<(), WalkerError> {
        self.work_queue().extend_cold([root]).await;
        self.drain().await
    }

    /// Process all queued directories until the queue is empty.
    ///
    /// Used by `circle.rs` in a loop: external work arriving via MPI is injected through `WorkQueue::extend_cold`, then
    /// this method is called again to process it.
    ///
    /// All open `Dir` file descriptors live in the queue on the heap, not on the async stack. Cancelling this future is
    /// safe; fds are dropped with the queue.
    pub async fn drain(&mut self) -> Result<(), WalkerError> {
        loop {
            let maybe_work = {
                let mut q = self.queue.lock().await;

                if let Some(de) = q.hot.last_mut() {
                    // Hot path: cached fd available - use openat to avoid full-path traversal.
                    let name = de.pending_subdirs.pop().unwrap();
                    let child_path = combine_paths(&de.dir_path, &name);
                    let raw_fd = de.fd.as_raw_fd();
                    // SAFETY: `de` is held in `q.hot` behind the `MutexGuard` and is not
                    // dropped until after `openat` returns. The fd is valid for the duration
                    // of this single synchronous syscall.
                    let result = Dir::openat(
                        unsafe { BorrowedFd::borrow_raw(raw_fd) },
                        name.as_c_str(),
                        OFlag::O_DIRECTORY | OFlag::O_RDONLY,
                        Mode::empty(),
                    );
                    if de.pending_subdirs.is_empty() {
                        q.hot.pop(); // fd closed here if this was the last pending entry
                    }
                    Some((result, child_path))
                } else if let Some(cold_path) = q.cold.pop() {
                    // Cold path: no cached fd - open by full path.
                    let result = Dir::open(
                        cold_path.as_c_str(),
                        OFlag::O_DIRECTORY | OFlag::O_RDONLY,
                        Mode::empty(),
                    );
                    Some((result, cold_path))
                } else {
                    None
                }
            };

            let Some((open_result, dir_path)) = maybe_work else {
                break;
            };

            match open_result {
                Err(e) => warn!("Failed to open {:?}: {e}", dir_path),
                Ok(dir) => self.process_open_dir(dir, dir_path).await?,
            }

            // All fds live in the queue on the heap, so yielding here is always safe.
            yield_now().await;
        }
        Ok(())
    }

    async fn process_open_dir(&mut self, dir: Dir, dir_path: CString) -> Result<(), WalkerError> {
        let mut iter = dir.into_iter();
        let raw_fd = iter.as_raw_fd();
        // https://github.com/nix-rust/nix/issues/2669
        // SAFETY: `iter` owns the underlying fd and is not dropped until after the loop. `borrowed_fd` is only used
        // within this function before `iter` is moved.
        let borrowed_fd = unsafe { BorrowedFd::borrow_raw(raw_fd) };

        let mut pending_subdirs: Vec<CString> = Vec::new();

        loop {
            let entry = match iter.next() {
                None => break,
                Some(Err(e)) => {
                    warn!("Error reading entry in {:?}: {e}", dir_path);
                    continue;
                }
                Some(Ok(e)) => e,
            };

            if is_dot_or_dotdot(entry.file_name()) {
                continue;
            }

            let is_dir = match entry.file_type() {
                Some(Type::Directory) => true,
                Some(_) => false,
                None => {
                    // TODO: This is a degenerate case that means we're going to be doing more syscalls than we normally
                    // would. When we traverse into a filesystem like this, we should warn about it.
                    // Filesystem didn't provide a file type in the dirent; stat to determine.
                    // Everything that isn't confirmed to be a directory is treated as a file.
                    match stat::fstatat(
                        borrowed_fd,
                        entry.file_name(),
                        AtFlags::AT_SYMLINK_NOFOLLOW,
                    ) {
                        Ok(s) => {
                            (SFlag::S_IFMT & SFlag::from_bits_truncate(s.st_mode)) == SFlag::S_IFDIR
                        }
                        Err(e) => {
                            warn!("fstatat failed for {:?}: {e}", entry.file_name());
                            false
                        }
                    }
                }
            };

            if is_dir {
                (self.on_dir)(&borrowed_fd, &dir_path, &entry).map_err(WalkerError::Nix)?;
                pending_subdirs.push(entry.file_name().to_owned());
            } else {
                (self.on_file)(&borrowed_fd, &dir_path, &entry).map_err(WalkerError::Nix)?;
            }
        }

        if !pending_subdirs.is_empty() {
            self.queue.lock().await.hot.push(DirectoryEntries {
                fd: Box::new(iter),
                dir_path,
                pending_subdirs,
            });
        }

        Ok(())
    }
}

/// A cheaply cloneable handle to the walker's internal queue for MPI work-sharing.
#[derive(Clone)]
pub struct WorkQueue(Rc<Mutex<Queue>>);

impl WorkQueue {
    pub async fn is_empty(&self) -> bool {
        self.0.lock().await.is_empty()
    }

    pub async fn len(&self) -> usize {
        self.0.lock().await.len()
    }

    /// Atomically extract up to `count` work items as full paths. Returns an empty vec if
    /// the queue is empty or count is zero.
    pub async fn try_split_work(&self, count: usize) -> Vec<CString> {
        self.0.lock().await.split_work(count)
    }

    /// Inject full paths (e.g. received from another MPI rank) into the cold queue.
    pub async fn extend_cold(&self, paths: impl IntoIterator<Item = impl AsRef<Path>>) {
        let mut q = self.0.lock().await;
        for p in paths {
            let bytes = p.as_ref().as_os_str().as_bytes().to_vec();
            // SAFETY: Filesystem paths sourced from the kernel via readdir (or transferred
            // as serialized byte strings from other ranks) cannot contain null bytes.
            q.cold.push(unsafe { CString::from_vec_unchecked(bytes) });
        }
    }
}

/// Create a walker with no-op callbacks.
#[allow(clippy::type_complexity)]
pub fn nothing_walker() -> Walker<
    impl FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
    impl FnMut(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>,
> {
    Walker::new(|_, _, _| Ok(()), |_, _, _| Ok(()))
}

/// Combine a base directory path and an entry name into a child path without UTF-8 interpretation.
pub fn combine_paths(base: &CStr, entry: &CStr) -> CString {
    let base_bytes = base.to_bytes();
    let entry_bytes = entry.to_bytes();
    let mut v = Vec::with_capacity(base_bytes.len() + 1 + entry_bytes.len());
    v.extend_from_slice(base_bytes);
    v.push(b'/');
    v.extend_from_slice(entry_bytes);
    // SAFETY: `base` and `entry` are `CStr`s (no interior null bytes), and filesystem
    // paths from readdir cannot contain null bytes. b'/' is not null.
    unsafe { CString::from_vec_unchecked(v) }
}

fn is_dot_or_dotdot(name: &CStr) -> bool {
    let b = name.to_bytes();
    b == b"." || b == b".."
}

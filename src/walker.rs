use nix::{
    NixPath as _,
    dir::{Dir, Entry},
    fcntl::OFlag,
    sys::stat::Mode,
};
use smol::{future::yield_now, lock::Mutex};
use std::{
    ffi::{CStr, CString},
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd},
        unix::ffi::OsStrExt,
    },
    path::Path,
    sync::Arc,
};
use tracing::error;

/// The signature of the callback called on each entry in a directory. The callback has this signature because it is the
/// most information we can provide about the entry without performing additional syscalls.
/// The parameters are as follows:
/// 1. `&dyn AsFd`: An open file descriptor for the directory containing the entry. Used so that you can grab additional
///    information about the entry, if necessary, by using the `fstatat` or similar *at calls that can take a file
///    descriptor.
/// 2. `&CStr`: The path of the directory containing the entry.
/// 3. `&Entry`: The directory entry itself. The only information this contains is the file type, the file name, and
///    inode.
pub type OnEntryCallback = fn(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>;

pub struct Walker {
    // TODO: Consider if there is a better data structure for stashing directories in the queue.
    queue: Arc<Mutex<Vec<CString>>>,
    on_file_entry: Option<OnEntryCallback>,
    on_directory_entry: Option<OnEntryCallback>,
}

impl Walker {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            on_file_entry: None,
            on_directory_entry: None,
        }
    }

    pub fn set_on_file_entry(&mut self, callback: OnEntryCallback) {
        self.on_file_entry = Some(callback);
    }

    pub fn set_on_directory_entry(&mut self, callback: OnEntryCallback) {
        self.on_directory_entry = Some(callback);
    }

    pub fn get_queue(&self) -> Arc<Mutex<Vec<CString>>> {
        self.queue.clone()
    }

    pub fn queue_len(&self) -> usize {
        self.queue.lock_blocking().len()
    }

    pub async fn walk(&self, root: &Path) -> nix::Result<()> {
        self.queue
            .lock()
            .await
            .push(CString::new(root.as_os_str().as_bytes()).expect("Path contained null byte!"));
        self.work_directory_queue().await
    }

    pub async fn push_new<I: IntoIterator<Item = impl AsRef<Path>>>(
        &self,
        paths: I,
    ) -> nix::Result<()> {
        let strings = paths.into_iter().map(|path| {
            CString::new(path.as_ref().as_os_str().as_bytes()).expect("Path contained null byte!")
        });
        self.queue.lock().await.extend(strings);
        Ok(())
    }

    /// Work on the current directory queue until it's empty, processing all available directories,
    /// calling `self.on_directory_entry` and `self.on_file_entry` as appropriate.
    pub async fn work_directory_queue(&self) -> nix::Result<()> {
        loop {
            // Pop a directory from the queue, or break if there's no work to do.
            let Some(dir_path) = self.queue.lock().await.pop() else {
                break;
            };
            match self.walk_directory_path(&dir_path) {
                Ok(None) => continue,
                Err(e) => error!("Error walking directory {:?}: {:?}", dir_path, e),
                Ok(Some((dir_iterator, entry, new_dir_path))) => {
                    // We have some subdirectory we can work on, so we repeatedly process it until we run out of
                    // subdirectories.

                    // We stash the dir iterator behind a `dyn AsRawFd` to keep it around, even when we use a borrowed
                    // file descriptor to prevent use-after-free issues.
                    let mut dir_iterator: Box<dyn AsRawFd> = Box::new(dir_iterator);
                    let mut current_entry = entry;
                    let mut current_dir_path = new_dir_path;
                    loop {
                        // SAFETY: We can never use `dir_fd` without after we drop `dir_iterator`
                        let dir_fd = unsafe { BorrowedFd::borrow_raw(dir_iterator.as_raw_fd()) };
                        match self.walk_directory_fd(dir_fd, &current_entry, &current_dir_path) {
                            Ok(None) => break,
                            Err(e) => {
                                error!("Error walking directory {:?}: {:?}", current_dir_path, e);
                                break;
                            }
                            Ok(Some((new_dir_iterator, entry, new_dir_path))) => {
                                dir_iterator = Box::new(new_dir_iterator);
                                current_entry = entry;
                                current_dir_path = new_dir_path;
                            }
                        }
                        // Allow other tasks to make progress
                        // TODO: There are multiple ways that we could yield and it's unclear which way would be better.
                        // 1. We can yield after each directory is processed
                        // 2. We can yield after processing all of the nested directories we have a fd for, this would
                        //    essentially be yielding after a depth-first traversal of the stack
                        // 3. We can yield after some number of directories
                        // 4. We can choose to never yield. This would probably break the multi-processing that we're
                        //    looking to achieve, but is technically possible.
                        yield_now().await;
                    }
                }
            }
            yield_now().await;
        }
        Ok(())
    }

    fn walk_directory_path(
        &self,
        dir_path: &CStr,
    ) -> nix::Result<Option<(impl AsRawFd, CString, CString)>> {
        let dir = nix::dir::Dir::open(
            dir_path,
            OFlag::O_DIRECTORY | OFlag::O_RDONLY,
            Mode::empty(),
        )?;
        self.on_directory(dir, dir_path)
    }

    // TODO: `walk_directory_fd` puts a lifetime on `AsRawFd` tying it to the lifetime of `self`  via
    // `impl AsRawFd + '_`, but this isn't really necessary, as `openat` shouldn't hold onto the fd we open it with.
    // Figure out how to remove or separate the lifetimes here.
    fn walk_directory_fd(
        &self,
        parent_fd: impl AsFd,
        dir_name: &CStr,
        dir_path: &CStr,
    ) -> nix::Result<Option<(impl AsRawFd + '_, CString, CString)>> {
        let dir = nix::dir::Dir::openat(
            parent_fd,
            dir_name,
            OFlag::O_DIRECTORY | OFlag::O_RDONLY,
            Mode::empty(),
        )?;
        self.on_directory(dir, dir_path)
    }

    /// Walk the provided `Dir`, iterating over its entries and calling the appropriate callbacks on each. Entry. If
    /// there are any subdirectories, we stash all but one of them in the queue for later processing, and return the
    /// final directory, the parent file descriptor, and the path of the directory for later processing.
    fn on_directory(
        &self,
        dir: Dir,
        dir_path: &CStr,
    ) -> nix::Result<Option<(impl AsRawFd + '_, CString, CString)>> {
        let mut dir_iterator = dir.into_iter();
        let mut dir_names = Vec::new();

        while let Some(entry) = dir_iterator.next() {
            // TODO: Handle errors we can recover from here, and avoid aborting everything.
            // Also determine what errors we can recover from.
            let entry = entry?;
            // TODO: Is there a more direct way of comparing these?
            if entry.file_name().to_bytes() == b"." || entry.file_name().to_bytes() == b".." {
                continue;
            }
            // https://github.com/nix-rust/nix/issues/2669
            // TODO: SAFETY: We borrow the file descriptor, but have to ensure that the dir_iterator is not dropped
            // while the borrowed file descriptor is still in use. I'm not 100% sure on the soundness of this without
            // consulting others.
            let fd = unsafe { BorrowedFd::borrow_raw(dir_iterator.as_raw_fd()) };
            match entry.file_type() {
                None => todo!(
                    "Need to stat the entry to determine the file type, since it wasn't provided by the filesystem"
                ),
                Some(nix::dir::Type::Directory) => {
                    if let Some(on_dir_entry) = &self.on_directory_entry {
                        on_dir_entry(&fd, dir_path, &entry)?;
                    }
                    //
                    dir_names.push(entry.file_name().to_owned());
                }
                _ => {
                    if let Some(on_file_entry) = &self.on_file_entry {
                        on_file_entry(&fd, dir_path, &entry)?;
                    }
                }
            }
        }
        match dir_names.len() {
            // There's no directory to seed the next directory walk with, so there's nothing to do.
            0 => Ok(None),
            // There's exactly one directory, so we can yield it without stashing anything in the queue.
            1 => {
                let entry = dir_names.pop().unwrap();
                let new_dir_path = combine_paths(dir_path, &entry);
                Ok(Some((dir_iterator, entry, new_dir_path)))
            }
            // There's more than one directory, so we stash all but one in the queue for later processing
            _ => {
                for entry in dir_names.drain(..dir_names.len() - 1) {
                    let new_dir_path = combine_paths(dir_path, &entry);
                    self.queue.lock_blocking().push(new_dir_path);
                }
                let entry = dir_names.pop().unwrap();
                let new_dir_path = combine_paths(dir_path, &entry);
                Ok(Some((dir_iterator, entry, new_dir_path)))
            }
        }
    }
}

impl Default for Walker {
    fn default() -> Self {
        Self::new()
    }
}

fn combine_paths(base: &CStr, entry: &CStr) -> CString {
    // TODO: The goal here is to avoid interpreting the path as UTF-8, is there a better way to do this?
    // TODO: Considering where we're getting this from, it should always be a valid path, even if it's not
    // valid UTF-8, so we should be able to use the unchecked variants of this function.
    let mut combined = Vec::with_capacity(base.len() + entry.len() + 1);
    combined.extend_from_slice(base.to_bytes());
    combined.push(b'/');
    combined.extend_from_slice(entry.to_bytes());
    combined.push(0);
    CString::from_vec_with_nul(combined).expect("Combined path contained null byte!")
}

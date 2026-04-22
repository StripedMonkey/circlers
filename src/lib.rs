
use std::{
    ffi::{CStr, CString},
    os::fd::{AsFd, AsRawFd, BorrowedFd},
};

use ferrompi::Communicator;
use futures::{FutureExt as _, select};
use nix::{
    NixPath as _,
    dir::{Dir, Entry},
    fcntl::{AtFlags, OFlag},
    sys::stat::{Mode, fstatat},
};
use smol::{lock::Mutex, pin};

type OnEntryCallback = fn(&dyn AsFd, &CStr, &Entry) -> nix::Result<()>;

pub struct Circle {
    comm: Communicator,
    on_file_entry: Option<OnEntryCallback>,
    on_directory_entry: Option<OnEntryCallback>,
    work_queue: Mutex<Vec<CString>>,
}

fn on_directory(fd: &dyn AsFd, dir_path: &CStr, entry: &Entry) -> nix::Result<()> {
    println!("Directory: {:?} in {:?}", entry.file_name(), dir_path);
    Ok(())
}

fn on_file(fd: &dyn AsFd, dir_path: &CStr, entry: &Entry) -> nix::Result<()> {
    let stat = match fstatat(fd, entry.file_name(), AtFlags::AT_SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(e) => return Err(e),
    };
    let mode = stat.st_mode;
    println!(
        "File: {:?} (mode {:o} in {:?}",
        entry.file_name(),
        mode,
        dir_path
    );
    Ok(())
}

impl Circle {
    pub fn new(world: &Communicator) -> Result<Self, ferrompi::Error> {
        Ok(Circle {
            comm: world.duplicate()?,
            on_file_entry: Some(on_file),
            on_directory_entry: Some(on_directory),
            work_queue: Mutex::new(Vec::new()),
        })
    }

    pub async fn make_progress(&mut self) {
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
        let work_dir_queue = self.work_directory_queue().fuse();
        pin!(work_dir_queue);
        select! {
            res = work_dir_queue => {
                if let Err(e) = res {
                    eprintln!("Error walking directory queue: {:?}", e);
                }

            }
        }
    }

    pub async fn walk_files_with_seed(&mut self, seed: &str) -> std::io::Result<()> {
        self.work_queue
            .lock()
            .await
            .push(CString::new(seed).expect("Seed path contained null byte!"));
        self.make_progress().await;
        Ok(())
    }

    /// Work on the current directory queue until it's empty, processing all available directories,
    /// calling `self.on_directory_entry` and `self.on_file_entry` as appropriate.
    async fn work_directory_queue(&self) -> nix::Result<()> {
        loop {
            // Pop a directory from the queue, or break if there's no work to do.
            let Some(dir_path) = self.work_queue.lock().await.pop() else {
                break;
            };
            match self.walk_directory_path(&dir_path) {
                Ok(None) => continue,
                Err(e) => eprintln!("Error walking directory {:?}: {:?}", dir_path, e),
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
                                eprintln!(
                                    "Error walking directory {:?}: {:?}",
                                    current_dir_path, e
                                );
                                break;
                            }
                            Ok(Some((new_dir_iterator, entry, new_dir_path))) => {
                                dir_iterator = Box::new(new_dir_iterator);
                                current_entry = entry;
                                current_dir_path = new_dir_path;
                            }
                        }
                    }
                }
            }
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
                return Ok(Some((dir_iterator, entry, new_dir_path)));
            }
            // There's more than one directory, so we stash all but one in the queue for later processing
            _ => {
                for entry in dir_names.drain(..dir_names.len() - 1) {
                    let new_dir_path = combine_paths(dir_path, &entry);
                    self.work_queue.lock_blocking().push(new_dir_path);
                }
                let entry = dir_names.pop().unwrap();
                let new_dir_path = combine_paths(dir_path, &entry);
                return Ok(Some((dir_iterator, entry, new_dir_path)));
            }
        }
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

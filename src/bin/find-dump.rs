use circlers::walker::combine_paths;
use clap::Parser;
use nix::dir::Entry;
use nix::fcntl::AtFlags;
use nix::sys::stat;
use postcard::experimental::max_size::MaxSize;
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::io::Write;
use std::io::{self, BufWriter};
use std::os::fd::AsFd;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::{Level, debug, span};
use tracing_subscriber::EnvFilter;

use circlers::Circle;
use ferrompi::Mpi;

#[derive(Parser, Debug)]
struct Args {
    root_directory: String,
}

// TODO:
// 1. Gather statistics about the walk, i.e the number of directories, files, and total size of files, failures, timing
//    information, and log them to stdout.
// 2. Clean up the Circle API
//    a. There should be some more rational choices w.r.t callbacks and exposing the results (and errors) of the walk.
//    b. It's possible that we want to make the thread-local execution internal to Circle. Considering the dependence on
//       the MPI to be initialized as `Funneled`, this may not be ideal.
// 3. Add functionality to write results per-rank to a tmpfile somewhere, and gather the results at the end.
// 4. Come up with a test suite to benchmark the walker and the MPI portions, sweeping through different
//    parameterizations.
// 5. Add some debug scripts to characterize the filesystem in a way that makes it easy to reproduce for later benches
//    and testing.
// 6. Explore whether doing something like io_uring would be beneficial. It's unclear whether io_uring is designed for
//    metadata workloads like this in a way that be able to speed up touching a lot of files once, but considering our
//    bottleneck is 99% (TODO: Profile to actually determine this number) the overhead of making the stat/openat
//    syscalls, I'd like to experiment with it.

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct BinFileEntry {
    user_id: u32,
    group_id: u32,
    permissions: u32,
    modification_time: u64,
    file_path: CString,
}

impl PartialOrd for BinFileEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BinFileEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.file_path.cmp(&other.file_path)
    }
}

#[derive(Debug, Serialize, Deserialize, MaxSize)]
struct WalkStats {
    num_dirs: usize,
    num_files: usize,
    total_size: u64,
}

fn main() -> io::Result<()> {
    // TODO: https://stackoverflow.com/questions/73156434/tell-if-stdout-is-a-tty-using-mpi-in-c
    // let should_color = io::stdout().is_terminal();
    let should_color = false;
    tracing_subscriber::fmt()
        .with_ansi(should_color)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let mpi = Mpi::init_thread(ferrompi::ThreadLevel::Funneled).map_err(std::io::Error::other)?;
    let world = mpi.world();
    let span = span!(Level::INFO, "process", rank = world.rank(),);
    let _enter = span.enter();

    let topology = world.topology(&mpi).unwrap();
    if world.rank() == 0 {
        debug!("{topology}");
    }
    let tmpdir_path = PathBuf::from_str("./").unwrap();

    let mut output = BufWriter::new(std::fs::File::create(
        tmpdir_path.join(format!("{}.bin", world.rank())),
    )?);
    let (mut dir_count, mut file_count, mut total_size) = (0, 0, 0);

    // https://users.rust-lang.org/t/implementation-of-fnonce-is-not-general-enough/78006/4
    let on_file_entry = |_fd: &dyn AsFd, path: &CStr, entry: &Entry| {
        let data = stat::fstatat(_fd, entry.file_name(), AtFlags::AT_SYMLINK_NOFOLLOW)?;
        file_count += 1;
        total_size += data.st_size as u64;

        let file_path = combine_paths(path, entry.file_name());
        let file_entry = BinFileEntry {
            user_id: data.st_uid,
            group_id: data.st_gid,
            permissions: data.st_mode,
            modification_time: data.st_mtime as u64,
            file_path,
        };
        postcard::to_io(&file_entry, &mut output).unwrap();
        Ok(())
    };
    let on_dir_entry = |_fd: &dyn AsFd, _path: &CStr, _entry: &Entry| {
        dir_count += 1;
        Ok(())
    };
    let seed = if world.rank() == 0 {
        Some(args.root_directory.as_ref())
    } else {
        None
    };
    let mut circle = Circle::new(&world).unwrap();
    // TODO: Currently using a local executor so that I don't have to consider the thread safety of the MPI
    // Communicator. In the future, we should determine the executor (and number of spawned threads).
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        circle.start_walk(seed, on_file_entry, on_dir_entry).await?;
        Ok(())
    }));
    output.flush()?;
    world
        .barrier()
        .expect("Error waiting for all processes to finish walking");
    debug!("Finished walk, gathering stats");
    let walk_stats = WalkStats {
        num_dirs: dir_count,
        num_files: file_count,
        total_size,
    };
    // TODO: Considering the effort below to do a `gatherv` with postcard, it should be wrapped a little more nicely.
    // I did it just to see how it's done, but it's not worth the overhead here. Wrapping it up to make it more
    // ergonomic would probably help but it's worth realizing that at the end of the day, the reason it's as complicated
    // as it is is because `postcard` is using a variable-length wire format. We don't really need that.
    let walk_stats = postcard::to_allocvec(&walk_stats).unwrap();
    let postcard_size = <WalkStats as MaxSize>::POSTCARD_MAX_SIZE;
    let mut walk_results = vec![0u8; postcard_size * world.size() as usize];
    let recvcounts = vec![postcard_size as i32; world.size() as usize];
    let displs = (0..world.size())
        .map(|i| i * postcard_size as i32)
        .collect::<Vec<_>>();
    world
        .gatherv(&walk_stats, &mut walk_results, &recvcounts, &displs, 0)
        .unwrap();
    if world.rank() == 0 {
        let mut combined_stats = WalkStats {
            num_dirs: 0,
            num_files: 0,
            total_size: 0,
        };
        let mut individual_stats = Vec::new();
        for chunk in walk_results.chunks_exact(postcard_size) {
            let stats: WalkStats = postcard::from_bytes(chunk).unwrap();
            combined_stats.num_dirs += stats.num_dirs;
            combined_stats.num_files += stats.num_files;
            combined_stats.total_size += stats.total_size;
            individual_stats.push(stats);
        }
        debug!("Combined walk stats: {combined_stats:?}");
        let mut output = std::fs::File::create(tmpdir_path.join("combined_stats.txt"))?;
        writeln!(output, "Combined walk stats: {combined_stats:?}")?;
        writeln!(output, "All stats: {individual_stats:?}")?;
    }
    Ok(())
}

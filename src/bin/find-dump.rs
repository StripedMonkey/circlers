use clap::Parser;
use nix::dir::Entry;
use std::ffi::CStr;
use std::io;
use std::io::Write;
use std::os::fd::AsFd;
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

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
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
    let mut output = std::fs::File::create(format!("{}.out.txt", world.rank())).unwrap();
    let on_entry_output = &mut output;
    // https://users.rust-lang.org/t/implementation-of-fnonce-is-not-general-enough/78006/4
    let on_file_entry = |_fd: &dyn AsFd, path: &CStr, entry: &Entry| {
        // Handle file entry
        writeln!(
            on_entry_output,
            "{}/{}",
            path.to_string_lossy(),
            entry.file_name().to_string_lossy()
        )
        .unwrap();
        Ok(())
    };
    let on_dir_entry = |_fd: &dyn AsFd, _path: &CStr, _entry: &Entry| {
        // Handle directory entry
        Ok(())
    };
    let mut circle = Circle::new(&world).unwrap();
    // TODO: Currently using a local executor so that I don't have to consider the thread safety of the MPI
    // Communicator. In the future, we should determine the executor (and number of spawned threads).
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        circle
            .start_walk(
                Some(args.root_directory.as_ref()),
                on_file_entry,
                on_dir_entry,
            )
            .await;
        Ok(())
    }));
    output.flush().unwrap();
    world
        .barrier()
        .expect("Error waiting for all processes to finish walking");
    Ok(())
}

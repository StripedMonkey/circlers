use clap::Parser;
use std::io;
use tracing::{Level, debug, span, trace};
use tracing_subscriber::EnvFilter;

use circlers::Circle;
use ferrompi::Mpi;

#[derive(Parser, Debug)]
struct Args {
    root_directory: String,
}

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
    let mut circle = Circle::new(&world).unwrap();
    // TODO: Currently using a local executor so that I don't have to consider the thread safety of the MPI
    // Communicator. In the future, we should determine the executor (and number of spawned threads).
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        circle.start_walk(Some(args.root_directory.as_ref())).await;
        Ok(())
    }));

    Ok(())
}

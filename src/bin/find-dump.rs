use clap::Parser;
use std::{io, path::Path};

use circlers::Circle;
use ferrompi::Mpi;
use smol::Task;

#[derive(Parser, Debug)]
struct Args {
    root_directory: String,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mpi = Mpi::init_thread(ferrompi::ThreadLevel::Funneled)
        .map_err(|e| std::io::Error::new(io::ErrorKind::Other, e))?;
    let world = mpi.world();
    println!("Hello from rank {} of {}", world.rank(), world.size());

    let topology = world.topology(&mpi).unwrap();
    if world.rank() == 0 {
        println!("{topology}");
    }
    let mut circle = Circle::new(&world).unwrap();
    // TODO: Currently using a local executor so that I don't have to consider the thread safety of the MPI
    // Communicator. In the future, we should determine the executor (and number of spawned threads).
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        let _res = circle.walk_files_with_seed(&args.root_directory).await;
        Ok(())
    }));

    Ok(())
}

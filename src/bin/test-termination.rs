use circlers::Circle;
use ferrompi::Mpi;
use std::io;
use std::time::Instant;
use tracing::{Level, debug, info, span};

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    let mpi = Mpi::init_thread(ferrompi::ThreadLevel::Funneled).map_err(std::io::Error::other)?;
    let world = mpi.world();
    let span = span!(Level::INFO,"process", rank = world.rank(),);
    let _enter = span.enter();
    debug!("Rank {} of {} started", world.rank(), world.size());

    // Topology is collective; query on all ranks and print only on rank 0.
    let topology = world.topology(&mpi).unwrap();
    if world.rank() == 0 {
        debug!("{topology}");
    }

    let mut circle = Circle::new(&world, |_,_,_| Ok(()), |_,_,_| Ok(())).map_err(std::io::Error::other)?;
    let runtime = smol::LocalExecutor::new();

    let start = Instant::now();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        circle.start_walk(None).await;
        info!(
            "Rank {} walk completed in {:?}",
            world.rank(),
            start.elapsed()
        );
        Ok(())
    }));

    info!("Rank {} terminating", world.rank());
    Ok(())
}

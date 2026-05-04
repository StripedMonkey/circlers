use clap::Parser;
use std::io;
use tracing::trace;
use tracing_subscriber::EnvFilter;

use circlers::walker::Walker;

#[derive(Parser, Debug)]
struct Args {
    root_directory: String,
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let root = args.root_directory.as_ref();
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        let mut walker = Walker::new(
            |_fd, name, _entry| {
                trace!("File entry: {:?}", name);
                Ok(())
            },
            |_fd, name, _entry| {
                trace!("Directory entry: {:?}", name);
                Ok(())
            },
        );
        walker.walk(root).await?;
        Ok(())
    }));

    Ok(())
}

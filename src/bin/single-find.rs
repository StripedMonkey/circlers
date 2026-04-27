use clap::Parser;
use std::io;

use circlers::walker::Walker;

#[derive(Parser, Debug)]
struct Args {
    root_directory: String,
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    let args = Args::parse();
    let root = args.root_directory.as_ref();
    let runtime = smol::LocalExecutor::new();
    let _result = smol::block_on(runtime.run::<io::Result<()>>(async {
        let mut walker = Walker::new();
        walker.set_on_directory_entry(|_fd, name, _entry| {
            println!("Directory entry: {:?}", name);
            Ok(())
        });
        walker.set_on_file_entry(|_fd, name, _entry| {
            println!("File entry: {:?}", name);
            Ok(())
        });
        walker.walk(root).await?;
        Ok(())
    }));

    Ok(())
}

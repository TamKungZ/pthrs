use std::{env, process::ExitCode};

use pthrs::{FaissIvfFlatIndex, SearchOptions, SearchWorkspace};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        return Err(usage());
    };
    if matches!(path.as_str(), "-h" | "--help") {
        println!("{}", usage());
        return Ok(());
    }
    let mut query_id = None;
    let mut k = 8usize;
    let mut nprobe = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--query-id" => {
                query_id = Some(parse(args.next(), "--query-id")?);
            }
            "--k" => k = parse(args.next(), "--k")?,
            "--nprobe" => nprobe = Some(parse(args.next(), "--nprobe")?),
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            unknown => return Err(format!("unknown option: {unknown}\n\n{}", usage())),
        }
    }

    let index = FaissIvfFlatIndex::open(&path).map_err(|error| error.to_string())?;
    println!("file: {path}");
    println!("type: IndexIVFFlat");
    println!("metric: {:?}", index.metric());
    println!("dimension: {}", index.dimension());
    println!("vectors: {}", index.len());
    println!("lists: {}", index.nlist());
    println!("default nprobe: {}", index.default_nprobe());
    println!("trained: {}", index.is_trained());

    if let Some(id) = query_id {
        let index = index.load().map_err(|error| error.to_string())?;
        let query = index
            .reconstruct(id)
            .ok_or_else(|| format!("vector ID {id} does not exist"))?
            .to_vec();
        let mut workspace = SearchWorkspace::new();
        let options = SearchOptions {
            k,
            nprobe: nprobe.unwrap_or(index.default_nprobe()),
        };
        index
            .search_into(&query, options, &mut workspace)
            .map_err(|error| error.to_string())?;
        println!("neighbors for ID {id}:");
        for neighbor in workspace.neighbors() {
            println!("  id={} distance={}", neighbor.id, neighbor.distance);
        }
    }
    Ok(())
}

fn parse<T: std::str::FromStr>(value: Option<String>, option: &str) -> Result<T, String> {
    value
        .ok_or_else(|| format!("{option} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn usage() -> String {
    "Usage: pthrs-index-inspect <model.index> [--query-id ID] [--k N] [--nprobe N]".to_owned()
}

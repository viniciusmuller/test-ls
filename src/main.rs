mod indexer;
mod parser;
mod simple_parser;

use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs,
    time::{Duration, Instant},
};

use indexer::Index;
use project_name::server;
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    let args: Vec<String> = env::args().into_iter().collect::<Vec<_>>();
    let cwd = env::current_dir()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    let paths = &[args.get(1).unwrap_or_else(|| &cwd)]; // fs::read_dir(&args[1]).unwrap();

    for path in paths {
        let now = Instant::now();
        let mut results = walkdir(path.as_str());
        let total_elapsed = now.elapsed();

        results.sort_by(|(_, _, e1), (_, _, e2)| e2.cmp(e1));

        println!("Top 10 slowest files to parse:");
        for (path, _, duration) in results.iter().take(10) {
            println!("{:?} - {:.2?}", path, duration);
        }

        let all_modules_index = results.iter().map(|(_, m, _)| m).flatten();
        let total_functions = all_modules_index
            .clone()
            .fold(0, |acc, index| acc + index.module.functions.len());

        println!("finished indexing: {} files", results.len());
        println!("Indexed {} modules", all_modules_index.count());
        println!("Indexed {} functions", total_functions);
        println!("Total reading + parsing time: {:.2?}", total_elapsed);
    }

    server::start_server()?;
    Ok(())
}

fn walkdir(path: &str) -> Vec<(String, Vec<Index>, Duration)> {
    let mut file_paths = Vec::new();

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(extension) = path.extension().and_then(OsStr::to_str) {
                match extension {
                    "ex" | "exs" => file_paths.push(path.to_owned()),
                    _ => (),
                }
            }
        }
    }

    use rayon::prelude::*;

    file_paths
        .par_iter()
        .map(|path| {
            let contents =
                fs::read_to_string(path).expect("Should have been able to read the file");
            let now = Instant::now();
            let parser = simple_parser::Parser::new(contents, path.to_str().unwrap().to_owned());
            let result = parser.parse();
            let index = parser.index(&result);
            let elapsed = now.elapsed();
            (path.to_str().unwrap().to_owned(), index, elapsed)
        })
        .collect::<Vec<_>>()
}

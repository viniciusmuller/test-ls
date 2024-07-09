mod parser;

use std::{
    env,
    ffi::OsStr,
    fs,
    path::Path,
    time::{Duration, Instant},
};

use walkdir::WalkDir;

fn main() {
    let args: Vec<String> = env::args().collect();

    let paths = &[&args[1]]; // fs::read_dir(&args[1]).unwrap();
    let mut total_successes = 0;
    let mut total_failures = 0;

    for path in paths {
        let now = Instant::now();
        let (mut successes, failures) = walkdir(path.as_str());
        let total_elapsed = now.elapsed();

        successes.sort_by(|(_, e1), (_, e2)| e2.cmp(e1));

        // println!("Parsing {}", path.display());

        // for path in &failures {
        //     println!("failed to parse {:?}", path);
        // }

        // println!("Top 10 slowest files to parse:");
        // for path in successes.iter().take(10) {
        //     println!("{:?}", path);
        // }

        println!(
            "finished indexing: errors: {} successes: {}",
            failures.len(),
            successes.len()
        );

        total_successes += successes.len();
        total_failures += failures.len();

        println!("Total reading + parsing time: {:.2?}", total_elapsed);

        println!(
            "successes total size: {} bytes\n",
            std::mem::size_of_val(&*successes)
        );
    }

    println!("total successes: {}, total failures: {}", total_successes, total_failures);
}

fn walkdir(path: &str) -> (Vec<(String, Duration)>, Vec<String>) {
    let mut file_paths = Vec::new();

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(extension) = path.extension().and_then(OsStr::to_str) {
                match extension {
                    "ex" => file_paths.push(path.to_owned()),
                    _ => (),
                }
            }
        }
    }

    use rayon::prelude::*;

    let results = file_paths
        .par_iter()
        .map(|path| {
            let contents = fs::read_to_string(path).expect("Should have been able to read the file");
            let now = Instant::now();
            let result = parser::parse(&contents);
            let elapsed = now.elapsed();
            (path, result, elapsed)
        })
        .collect::<Vec<_>>();

    let successes = results
        .iter()
        .filter(|(_path, e, _)| e.is_ok())
        .map(|(path, _, elapsed)| (path.to_str().unwrap().to_owned(), elapsed.to_owned()))
        .collect::<Vec<_>>();

    let failures = results
        .into_iter()
        .filter(|(_path, e, _)| e.is_err())
        .map(|(path, _, _)| path.to_str().unwrap().to_owned())
        .collect();

    return (successes, failures);
}

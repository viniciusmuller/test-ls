mod completion_engine;
mod completion_engine_actor;
mod indexer;
mod parser;
mod simple_parser;

use std::sync::mpsc::channel;
use std::{env, error::Error, ffi::OsStr, fs, sync::mpsc::Sender, thread, time::Instant};

use completion_engine::{CompletionContext, CompletionQuery, GlobalIndexMessage};
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    env_logger::init();

    // TODO: run indexing and language server at the same time in different threads
    // TODO: get nicer log crate for logging things such as "finished parsing, etc"
    // use message passing or update global index directly through the thread + mutexes
    // TODO: move actual language server engine to an IO-isolated module to ease of testing

    let (tx, rx) = channel::<GlobalIndexMessage>();

    let index_handle = thread::spawn(|| indexer(tx));

    // let server_handle = thread::spawn(|| {
    //     server::start_server().unwrap();
    // });
    //
    let global_index = thread::spawn(|| {
        completion_engine_actor::start(rx);
    });

    index_handle.join().unwrap();
    global_index.join().unwrap();
    // server_handle.join().unwrap();

    Ok(())
}

fn indexer(tx: Sender<GlobalIndexMessage>) {
    let args: Vec<String> = env::args().into_iter().collect::<Vec<_>>();
    let cwd = env::current_dir()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    let paths = &[args.get(1).unwrap_or_else(|| &cwd)]; // fs::read_dir(&args[1]).unwrap();

    for path in paths {
        walkdir(tx.clone(), path.as_str());

        // results.sort_by(|(_, _, e1), (_, _, e2)| e2.cmp(e1));

        // println!("Top 10 slowest files to parse:");
        // for (path, _, duration) in results.iter().take(10) {
        //     println!("{:?} - {:.2?}", path, duration);
        // }

        // let all_modules_index = results.iter().map(|(_, m, _)| m).flatten();
        // let total_functions = all_modules_index
        //     .clone()
        //     .fold(0, |acc, index| acc + index.module.functions.len());

        // info!("finished indexing: {} files", results.len());
        // info!("Indexed {} modules", all_modules_index.count());
        // info!("Indexed {} functions", total_functions);
        // info!("Total reading + parsing time: {:.2?}", total_elapsed);
    }
}

fn walkdir(tx: Sender<GlobalIndexMessage>, path: &str) {
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

    let now = Instant::now();
    file_paths.par_iter().for_each(|path| {
        let contents = fs::read_to_string(path).expect("Should have been able to read the file");
        let parser = simple_parser::Parser::new(contents, path.to_str().unwrap().to_owned());
        let result = parser.parse();
        let indexes = parser.index(&result);

        for i in indexes {
            tx.send(GlobalIndexMessage::NewModule(i)).unwrap();
        }
    });
    let elapsed = now.elapsed();

    tx.send(GlobalIndexMessage::FinishedIndexing(elapsed))
        .unwrap();

    tx.send(GlobalIndexMessage::Query(CompletionQuery {
        query: "Ecto.Qu".to_owned(),
        context: CompletionContext::Module,
    }))
    .unwrap();

    tx.send(GlobalIndexMessage::Query(CompletionQuery {
        query: "Axon.M".to_owned(),
        context: CompletionContext::Module,
    }))
    .unwrap();

    tx.send(GlobalIndexMessage::Query(CompletionQuery {
        query: "map_".to_owned(),
        context: CompletionContext::Scope,
    }))
    .unwrap();
}

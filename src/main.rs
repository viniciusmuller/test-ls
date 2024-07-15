mod completion_engine;
mod completion_engine_actor;
mod indexer;
mod interner;
mod language_server_actor;
mod parser;
mod simple_parser;

use std::{env, error::Error, ffi::OsStr, fs, time::Instant};

use actix::{Actor, Addr, System};
use completion_engine_actor::{CompletionEngineActor, CompletionEngineMessage};
use walkdir::WalkDir;

#[actix::main]
async fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    env_logger::init();

    let completion_engine_addr = completion_engine_actor::CompletionEngineActor::default().start();
    // let _language_server_addr =
    //     language_server_actor::LanguageServerActor::new(completion_engine_addr.clone()).start();

    indexer(completion_engine_addr.clone()).await;

    language_server_actor::start_server(completion_engine_addr)
        .await
        .unwrap();

    // stop system and exit
    System::current().stop();

    // index_handle.join().unwrap();
    // completion_engine_actor.join().unwrap();
    // server_handle.join().unwrap();

    Ok(())
}

async fn indexer(completion_engine_actor: Addr<CompletionEngineActor>) {
    let args: Vec<String> = env::args().into_iter().collect::<Vec<_>>();
    let cwd = env::current_dir()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    let paths = &[args.get(1).unwrap_or_else(|| &cwd)]; // fs::read_dir(&args[1]).unwrap();

    for path in paths {
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
            let contents =
                fs::read_to_string(path).expect("Should have been able to read the file");
            let parser = simple_parser::Parser::new(contents, path.to_str().unwrap().to_owned());
            let result = parser.parse();
            let indexes = parser.index(&result);

            for i in indexes {
                completion_engine_actor.do_send(CompletionEngineMessage::NewModule(i))
            }
        });
        let elapsed = now.elapsed();

        let _ = completion_engine_actor
            .send(CompletionEngineMessage::FinishedIndexing(elapsed))
            .await;

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

// fn index_dir_recursive(completion_engine_tx: Sender<CompletionEngineMessage>, path: &str) {

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "Ecto.Qu".to_owned(),
//     //         context: CompletionContext::Module,
//     //     }))
//     //     .unwrap();

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "Axon.M".to_owned(),
//     //         context: CompletionContext::Module,
//     //     }))
//     //     .unwrap();

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "ma".to_owned(),
//     //         context: CompletionContext::ModuleContents(interner::get_string("Enum").unwrap()),
//     //     }))
//     //     .unwrap();

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "i".to_owned(),
//     //         context: CompletionContext::ModuleContents(interner::get_string("IO").unwrap()),
//     //     }))
//     //     .unwrap();

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "A".to_owned(),
//     //         context: CompletionContext::Module,
//     //     }))
//     //     .unwrap();

//     // completion_engine_tx
//     //     .send(GlobalIndexMessage::Query(CompletionQuery {
//     //         query: "ExDoc".to_owned(),
//     //         context: CompletionContext::Module,
//     //     }))
//     //     .unwrap();
// }

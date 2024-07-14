use std::time::Instant;
use std::{collections::HashMap, sync::mpsc::Receiver};

use log::info;

use crate::completion_engine::{CompletionEngine, CompletionQuery, GlobalIndexMessage};

pub fn start(rx: Receiver<GlobalIndexMessage>) {
    let mut engine = CompletionEngine::new();

    while let Ok(msg) = rx.recv() {
        match msg {
            GlobalIndexMessage::NewModule(i) => {
                engine.add_module(i)
            }
            GlobalIndexMessage::FinishedIndexing(total_time) => {
                info!("Finished indexing in {:.2?}", total_time);
            }
            GlobalIndexMessage::Query(query) => {
                engine.query(query);
            }
        }
    }
}

use std::sync::mpsc::Receiver;
use std::sync::{Arc, RwLock};

use log::info;
use string_interner::backend::StringBackend;
use string_interner::StringInterner;

use crate::completion_engine::{CompletionEngine, GlobalIndexMessage};

pub fn start(rx: Receiver<GlobalIndexMessage>) {
    let mut engine = CompletionEngine::new();

    while let Ok(msg) = rx.recv() {
        match msg {
            GlobalIndexMessage::NewModule(i) => engine.add_module(i),
            GlobalIndexMessage::FinishedIndexing(total_time) => {
                engine.finished_indexing(total_time);
            }
            GlobalIndexMessage::Query(query) => {
                engine.query(query);
            }
        }
    }
}

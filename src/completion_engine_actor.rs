use std::time::Duration;

use crate::{
    completion_engine::{CompletionEngine, CompletionQuery},
    indexer::Index,
};

use actix::prelude::*;

use crate::completion_engine::CompletionItem;

#[derive(Message)]
#[rtype(result = "Vec<CompletionItem>")]
pub enum CompletionEngineMessage {
    NewModule(Index),
    FinishedIndexing(Duration),
    Query(CompletionQuery),
}

pub struct CompletionEngineActor {
    completion_engine: CompletionEngine,
}

impl Actor for CompletionEngineActor {
    type Context = Context<Self>;
}

impl Handler<CompletionEngineMessage> for CompletionEngineActor {
    type Result = Vec<CompletionItem>;

    fn handle(&mut self, msg: CompletionEngineMessage, _ctx: &mut Self::Context) -> Self::Result {
        match msg {
            CompletionEngineMessage::NewModule(i) => {
                self.completion_engine.add_module(i);
                vec![]
            }
            CompletionEngineMessage::FinishedIndexing(total_time) => {
                self.completion_engine.finished_indexing(total_time);
                vec![]
            }
            CompletionEngineMessage::Query(query) => self.completion_engine.query(query),
        }
    }
}

impl Default for CompletionEngineActor {
    fn default() -> Self {
        Self {
            completion_engine: CompletionEngine::new(),
        }
    }
}

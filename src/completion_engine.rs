use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use log::info;
use string_interner::{DefaultBackend, StringInterner};

use crate::indexer::Index;

pub enum GlobalIndexMessage {
    NewModule(Index),
    FinishedIndexing(Duration),
    Query(CompletionQuery),
}

pub struct CompletionEngine {
    interner: Arc<RwLock<StringInterner<DefaultBackend>>>,
    modules_ids_index: Vec<String>,
    modules_index: HashMap<usize, Index>,
}

#[derive(Debug)]
pub enum CompletionContext {
    Module,
    ModuleContents,
    Scope,
}

#[derive(Debug)]
pub struct CompletionQuery {
    pub query: String,
    pub context: CompletionContext,
}

pub enum CompletionKind {
    Module,
    Name,
}

#[derive(Debug)]
pub enum CompletionItem {
    Module(String),
    ModuleFunction(String),
}

impl CompletionEngine {
    pub fn new(interner: Arc<RwLock<StringInterner<DefaultBackend>>>) -> Self {
        Self {
            interner,
            modules_index: HashMap::new(),
            modules_ids_index: vec![],
        }
    }

    // TODO: will need current scope here
    // TODO: function to get current scope
    // TODO: handle filenames (we need to tie modules to filenames)
    pub fn query(&self, query: CompletionQuery) -> Vec<CompletionItem> {
        info!("Got query: {:?}", query);

        let now = Instant::now();
        let query_result = match query.context {
            CompletionContext::Module => self
                .modules_index
                .iter()
                .filter(|(k, _)| {
                    let index = k.to_owned().to_owned();
                    let name = &self.modules_ids_index[index];
                    name.starts_with(&query.query)
                })
                .take(10)
                .map(|(k, _)| {
                    let index = k.to_owned().to_owned();
                    let name = &self.modules_ids_index[index];
                    CompletionItem::Module(name.to_string())
                })
                .collect::<Vec<_>>(),
            CompletionContext::ModuleContents => todo!(),
            CompletionContext::Scope => todo!(),
        };
        let elapsed = now.elapsed();

        info!("Query result: {:?}", query_result);
        info!("Query finished in {:.2?}", elapsed);
        query_result
    }

    pub fn add_module(&mut self, index: Index) {
        let module_id = self.modules_ids_index.len();
        self.modules_ids_index.push(index.module.name.clone());
        self.modules_index.insert(module_id, index);
    }
}

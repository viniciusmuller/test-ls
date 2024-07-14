use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use log::trace;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

use crate::{indexer::Index, interner::{get_string, resolve_string}};

pub enum GlobalIndexMessage {
    NewModule(Index),
    FinishedIndexing(Duration),
    Query(CompletionQuery),
}

pub struct CompletionEngine {
    modules_index: HashMap<DefaultSymbol, Index>,
}

#[derive(Debug)]
pub enum CompletionContext {
    Module,
    ModuleContents(DefaultSymbol),
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
    Function(String),
    ModuleFunction(String),
}

impl CompletionEngine {
    pub fn new() -> Self {
        Self {
            modules_index: HashMap::new(),
        }
    }

    // TODO: will need current scope here
    // TODO: function to get current scope
    pub fn query(&self, query: CompletionQuery) -> Vec<CompletionItem> {
        trace!("Got query: {:?}", query);

        let now = Instant::now();
        let query_result = match query.context {
            CompletionContext::Module => self
                .modules_index
                .iter()
                .filter(|(s, _)| {
                    let module_name = resolve_string(**s).unwrap();
                    module_name.starts_with(&query.query)
                })
                .take(10)
                .map(|(s, _)| {
                    let module_name = resolve_string(*s).unwrap();
                    CompletionItem::Module(module_name.to_string())
                })
                .collect::<Vec<_>>(),
            CompletionContext::ModuleContents(module_name) => {
                match self.modules_index.get(&module_name) {
                    Some(index) => index
                        .module
                        .functions
                        .iter()
                        .filter(|f| f.name.starts_with(&query.query))
                        .take(10)
                        .map(|f| CompletionItem::Function(f.name.to_owned()))
                        .collect(),
                    None => vec![],
                }
            }
            CompletionContext::Scope => todo!(),
        };
        let elapsed = now.elapsed();

        trace!("Query result: {:?}", query_result);
        trace!("Query finished in {:.2?}", elapsed);
        query_result
    }

    pub fn add_module(&mut self, index: Index) {
        self.modules_index.insert(index.module.name, index);
    }
}

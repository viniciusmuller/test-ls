use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use log::info;

use crate::indexer::Index;

pub enum GlobalIndexMessage {
    NewModule(Index),
    FinishedIndexing(Duration),
    Query(CompletionQuery),
}

pub struct CompletionEngine {
    modules_ids_index: Vec<String>,
    modules_index: HashMap<usize, Index>,
    functions_index: HashMap<String, usize>,
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
    pub fn new() -> Self {
        Self {
            modules_index: HashMap::new(),
            functions_index: HashMap::new(),
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
            CompletionContext::Scope => self
                .functions_index
                .iter()
                .filter(|(k, _)| k.starts_with(&query.query))
                .take(10)
                .map(|(k, v)| {
                    let index = v.to_owned().to_owned();
                    let module_name = &self.modules_ids_index[index];
                    let result = format!("{}.{}", module_name, k);
                    CompletionItem::ModuleFunction(result)
                })
                .collect::<Vec<_>>(),
        };
        let elapsed = now.elapsed();

        info!("Query result: {:?}", query_result);
        info!("Query finished in {:.2?}", elapsed);
        query_result
    }

    pub fn add_module(&mut self, index: Index) {
        let module_id = self.modules_ids_index.len();
        self.modules_ids_index.push(index.module.name.clone());

        for function in &index.module.functions {
            if !function.is_private {
                self.functions_index
                    .insert(function.name.clone(), module_id.clone());
            }
        }

        self.modules_index.insert(module_id, index);
    }
}

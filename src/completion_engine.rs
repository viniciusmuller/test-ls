use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use log::{info, trace};
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};
use trie_rs::{Trie, TrieBuilder};

use crate::{
    indexer::Index,
    interner::{get_string, resolve_string},
};

pub enum GlobalIndexMessage {
    NewModule(Index),
    FinishedIndexing(Duration),
    Query(CompletionQuery),
}

pub struct CompletionEngine {
    modules_index: HashMap<DefaultSymbol, Index>,
    modules_trie_builder: TrieBuilder<u8>,
    modules_trie: Option<Trie<u8>>,
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
            modules_trie_builder: TrieBuilder::new(),
            modules_trie: None,
        }
    }

    // TODO: will need current scope here
    // TODO: function to get current scope
    pub fn query(&self, query: CompletionQuery) -> Vec<CompletionItem> {
        trace!("Got query: {:?}", query);

        let now = Instant::now();
        let query_result = match query.context {
            CompletionContext::Module => match self.modules_trie {
                None => vec![],
                Some(ref trie) => {
                    let result: Vec<String> = trie.predictive_search(&query.query).collect();
                    result
                        .into_iter()
                        .map(|a| CompletionItem::Module(a))
                        .collect()
                }
            },
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
        let module_name = resolve_string(index.module.name).unwrap();
        self.modules_trie_builder.push(module_name);
        self.modules_index.insert(index.module.name, index);
    }

    pub fn finished_indexing(&mut self, total_time: Duration) {
        self.modules_trie = Some(self.modules_trie_builder.clone().build());
        info!("Finished indexing in {:.2?}", total_time);
    }
}

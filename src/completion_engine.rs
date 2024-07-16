use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use string_interner::DefaultSymbol;
use tracing::{info, trace};
use trie_rs::{Trie, TrieBuilder};

use crate::{
    indexer::Index,
    interner::{get_string, resolve_string},
};

pub struct CompletionEngine {
    modules_index: HashMap<DefaultSymbol, Index>,
    modules_trie_builder: TrieBuilder<u8>,
    modules_trie: Option<Trie<u8>>,
}

#[derive(Debug)]
pub enum CursorContext {
    Dot(DotContext),
    Module(String),
    Identifier(String),
    Unknown,
}

/// When we find the cursor position in a dot, this represents the left hand side of the dot.
#[derive(Debug)]
pub enum DotContext {
    Module(String, String),
    Identifier(String, String),
    Other(String, String),
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
    pub fn query(&self, ctx: CursorContext) -> Vec<CompletionItem> {
        trace!("Got query: {:?}", ctx);

        let now = Instant::now();
        let vec = match ctx {
            CursorContext::Module(module) => match self.modules_trie {
                None => vec![],
                Some(ref trie) => {
                    let result: Vec<CompletionItem> = trie
                        .predictive_search(&module)
                        .into_iter()
                        .take(10)
                        .map(CompletionItem::Module)
                        .collect();
                    result
                }
            },
            CursorContext::Dot(DotContext::Module(module_name, right)) => get_string(&module_name)
                .and_then(|sym| self.modules_index.get(&sym))
                .map(|index| {
                    index
                        .module
                        .functions
                        .iter()
                        .filter(|f| f.name.starts_with(&right))
                        .take(10)
                        .map(|f| CompletionItem::Function(f.name.to_owned()))
                        .collect()
                })
                .unwrap_or(vec![]),
            CursorContext::Dot(_) => vec![],
            CursorContext::Identifier(_) => todo!(),
            CursorContext::Unknown => todo!(),
        };
        let query_result = vec;
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

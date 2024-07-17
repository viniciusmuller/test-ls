use std::collections::HashMap;

use tree_sitter::Tree;

/// Represents the document store, which is used by the language server process
/// to keep track of tree-sitter ASTs for currently open buffers.
struct DocumentStore<'a> {
    store: HashMap<String, &'a Tree>,
}

impl<'a> DocumentStore<'a> {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn get_document_tree(&self, path: &str) -> Option<&'a Tree> {
        self.store.get(path).copied()
    }

    /// https://docs.rs/tree-sitter/0.22.6/tree_sitter/#editing
    pub fn upsert_document(&mut self, path: &str) -> Option<&'a Tree> {
        match self.store.get(path) {
            Some(_existing_tree) => {
                // Edit existing tree and use it in order to parse new one
                todo!()
            }
            None => {
                // Parse from scratch and insert into tree
                todo!()
            }
        }
    }
}

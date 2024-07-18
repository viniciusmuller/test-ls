use std::collections::HashMap;

use tree_sitter::Tree;

use crate::simple_parser;

/// Represents the document store, which is used by the language server process
/// to keep track of tree-sitter ASTs for currently open buffers.
#[derive(Debug)]
pub struct DocumentStore {
    store: HashMap<String, Tree>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn get_document_tree(&self, path: &str) -> Option<Tree> {
        self.store.get(path).cloned()
    }

    fn store_document_tree(&mut self, path: String, tree: &Tree) -> Option<Tree> {
        self.store.insert(path, tree.clone())
    }

    // NOTE: currently parsing every keystroke works fine, even for a file with ~20K lines it runs
    // in a few milliseconds
    /// https://docs.rs/tree-sitter/0.22.6/tree_sitter/#editing
    pub fn upsert_document(&mut self, path: String, content: &str) -> Option<Tree> {
        match self.store.get(&path) {
            Some(_existing_tree) => {
                // TODO: Edit existing tree and use it in order to parse new one
                let tree = simple_parser::build_tree(content);
                self.store_document_tree(path, &tree);
                Some(tree)
            }
            None => {
                // Parse from scratch and insert into tree
                let tree = simple_parser::build_tree(content);
                self.store_document_tree(path, &tree);
                Some(tree)
            }
        }
    }
}

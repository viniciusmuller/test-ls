use std::collections::HashMap;

use crate::simple_parser::{Expression, Module};

#[derive(Debug)]
pub struct FunctionIndex {}

#[derive(Debug)]
pub struct ModuleIndex {
    name: String,
    docs: Option<String>,
    scope: ScopeIndex,
    attributes: HashMap<String, Box<Expression>>,
    functions: HashMap<String, FunctionIndex>,
    macros: Vec<String>,
}

impl Default for ModuleIndex {
    fn default() -> Self {
        Self {
            docs: None,
            scope: ScopeIndex::default(),
            attributes: HashMap::new(),
            functions: HashMap::new(),
            macros: vec![],
            name: "MODULE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub struct ScopeIndex {
    parent: Option<Box<ScopeIndex>>,
    imports: Vec<String>,
    aliases: Vec<String>,
    requires: Vec<String>,
    variables: Vec<String>,
    modules: HashMap<String, ModuleIndex>,
}

impl Default for ScopeIndex {
    fn default() -> Self {
        Self {
            parent: None,
            imports: vec![],
            aliases: vec![],
            requires: vec![],
            variables: vec![],
            modules: HashMap::new(),
        }
    }
}

pub fn build_module_index(module: &Module) -> Option<ModuleIndex> {
    let a = module.body.as_ref();

    let mut index = ModuleIndex {
        name: module.name.to_owned(),
        ..ModuleIndex::default()
    };

    match a {
        Expression::Scope(scope) => {
            build_module_body_index(&scope.body, &mut index);
            Some(index)
        }
        _ => None,
    }
}

fn build_module_body_index(expressions: &Vec<Expression>, index: &mut ModuleIndex) {
    for expression in expressions {
        match expression {
            Expression::Module(module) => {
                let boxed = Box::new(module);

                if let Some(indexed_module) = build_module_index(boxed.as_ref()) {
                    index
                        .scope
                        .modules
                        .insert(module.name.to_owned(), indexed_module);
                }
            }
            Expression::FunctionDef(function) => {
                index
                    .functions
                    .insert(function.name.to_owned(), FunctionIndex {});
            }
            Expression::Attribute(name, body) => {
                index.attributes.insert(name.to_owned(), body.clone());
            }

            // TODO: apparently the scope inside (a;b) blocks leak to the outer scope in the compiler
            // Expression::Scope(_) => todo!(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn index_module() {
        assert_eq!(true, false)
    }
}

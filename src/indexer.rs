use std::collections::HashMap;

use crate::simple_parser::{Expression, Module};

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionIndex {
    is_private: bool,
    name: String,
    doc: Option<String>,
    spec: Option<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleIndex {
    name: String,
    docs: Option<String>,
    scope: ScopeIndex,
    attributes: Vec<(String, Expression)>,
    types: Vec<(String, Expression)>,
    functions: Vec<FunctionIndex>,
    macros: Vec<String>,
}

impl Default for ModuleIndex {
    fn default() -> Self {
        Self {
            docs: None,
            scope: ScopeIndex::default(),
            attributes: vec![],
            functions: vec![],
            types: vec![],
            macros: vec![],
            name: "MODULE".to_owned(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
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
    let mut index = ModuleIndex {
        name: module.name.to_owned(),
        ..ModuleIndex::default()
    };

    match module.body.as_ref() {
        Expression::Scope(scope) => {
            build_module_body_index(&scope.body, &mut index);
            Some(index)
        }
        _ => None,
    }
}

fn build_module_body_index(expressions: &Vec<Expression>, index: &mut ModuleIndex) {
    let mut last_doc: Option<String> = None;
    let mut last_spec: Option<Expression> = None;

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
                index.functions.push(FunctionIndex {
                    is_private: function.is_private,
                    name: function.name.to_owned(),
                    doc: last_doc.clone(),
                    spec: last_spec.clone(),
                });

                if last_doc.is_some() {
                    last_doc = None
                };

                if last_spec.is_some() {
                    last_spec = None;
                };
            }
            Expression::Attribute(name, body) => match name.as_str() {
                "doc" => {
                    if let Expression::String(s) = body.as_ref() {
                        last_doc = Some(s.to_owned());
                    };
                }
                "spec" => {
                    last_spec = Some(*body.to_owned());
                }
                "type" => {
                    index.types.push((name.to_owned(), *body.to_owned()));
                }
                "moduledoc" => {
                    if let Expression::String(s) = body.as_ref() {
                        index.docs = Some(s.to_owned());
                    };
                }
                _ => index.attributes.push((name.to_owned(), *body.clone())),
            },

            // TODO: apparently the scope inside (a;b) blocks leak to the outer scope in the compiler
            // Expression::Scope(_) => todo!(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        indexer::FunctionIndex,
        simple_parser::{Expression, Parser},
    };

    use super::{build_module_index, ModuleIndex};

    fn index(code: &str) -> ModuleIndex {
        let parser = Parser::new(code.to_owned(), "nofile".to_owned());
        if let Expression::Module(m) = parser.parse() {
            build_module_index(&m).unwrap()
        } else {
            panic!("failed to parse module")
        }
    }

    #[test]
    fn index_simple_module() {
        let code = r#"
        defmodule MyModule do
            @moduledoc "this is a nice module!"

            def func(a, b) do
                a + b
            end
        end
        "#;
        let result = index(code);
        let target = ModuleIndex {
            name: "MyModule".to_owned(),
            docs: Some("this is a nice module!".to_owned()),
            functions: vec![FunctionIndex {
                is_private: false,
                name: "func".to_owned(),
                doc: None,
                spec: None,
            }],
            ..ModuleIndex::default()
        };

        assert_eq!(result, target)
    }

    #[test]
    fn index_function_docs_and_specs() {
        let code = r#"
        defmodule MyModule do
            @doc "sums two numbers"
            @spec func(integer(), integer()) :: integer()
            defp func(a, b) do
                a + b
            end

            defp no_docs(_) do

            end
        end
        "#;
        let result = index(code);
        let target = ModuleIndex {
            name: "MyModule".to_owned(),
            docs: None,
            functions: vec![
                FunctionIndex {
                    is_private: true,
                    name: "func".to_owned(),
                    doc: Some("sums two numbers".to_string()),
                    spec: None,
                },
                FunctionIndex {
                    is_private: true,
                    name: "no_docs".to_owned(),
                    doc: None,
                    spec: None,
                },
            ],
            ..ModuleIndex::default()
        };

        assert_eq!(result, target)
    }
}

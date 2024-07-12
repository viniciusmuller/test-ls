use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::simple_parser::{BinaryOperator, Expression, Module, Point, Scope};

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionIndex {
    location: Point,
    is_private: bool,
    name: String,
    doc: Option<String>,
    spec: Option<Expression>,
    scope: Rc<RefCell<ScopeIndex>>,
    // TODO: probably will need to store function body to be able to compute lambda scopes
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleIndex {
    // location: Point,
    name: String,
    docs: Option<String>,
    scope: Rc<RefCell<ScopeIndex>>,
    attributes: Vec<(String, Expression)>,
    types: Vec<(String, Expression)>,
    // TODO: group function clauses into a single function structure by name and arity
    functions: Vec<FunctionIndex>,
    macros: Vec<String>,
}

impl Default for ModuleIndex {
    fn default() -> Self {
        Self {
            // location: Point { line: 1, column: 0 },
            docs: None,
            scope: Rc::new(RefCell::new(ScopeIndex::default())),
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
    parent: Option<Rc<RefCell<ScopeIndex>>>,
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
                    let mut scope = index.scope.borrow_mut();
                    scope.modules.insert(module.name.to_owned(), indexed_module);
                }
            }
            Expression::BinaryOperation(operation) => {
                if operation.operator == BinaryOperator::Match {
                    let left_ids = extract_identifiers(&operation.left);
                    let mut scope = index.scope.borrow_mut();
                    scope.variables.extend(left_ids)
                }
            }
            Expression::FunctionDef(function) => {
                index.functions.push(FunctionIndex {
                    location: function.location.clone(),
                    is_private: function.is_private,
                    name: function.name.to_owned(),
                    doc: last_doc.clone(),
                    spec: last_spec.clone(),
                    scope: Rc::new(RefCell::new(build_function_scope_index(
                        &function.body,
                        Some(index.scope.clone()),
                    ))),
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

fn extract_identifiers(ast: &Expression) -> Vec<String> {
    let mut buffer = Vec::new();
    do_extract_identifiers(ast, &mut buffer);
    buffer
}

fn do_extract_identifiers(ast: &Expression, buffer: &mut Vec<String>) {
    match ast {
        // TODO: update here when we support lists, tuples, maps, and others
        Expression::BinaryOperation(operation) => {
            do_extract_identifiers(&operation.left, buffer);
            do_extract_identifiers(&operation.right, buffer);
        }
        Expression::Identifier(id) => buffer.push(id.to_string()),
        // NOTE: we don't do it recursively for the Unparsed case because we might
        // end up indexing a lot of identifiers that are not variables by mistake (e.g: function call name)
        _ => {}
    }
}

fn build_function_scope_index(
    ast: &Expression,
    parent: Option<Rc<RefCell<ScopeIndex>>>,
) -> ScopeIndex {
    let mut scope = ScopeIndex {
        parent,
        ..ScopeIndex::default()
    };

    match ast {
        Expression::Scope(Scope { body }) => {
            for expression in body {
                parse_scope_expression(&expression, &mut scope);
            }
        }
        _ => {}
    };

    scope
}

fn parse_scope_expression(expression: &Expression, scope: &mut ScopeIndex) {
    match expression {
        Expression::Module(module) => {
            let boxed = Box::new(module);

            if let Some(indexed_module) = build_module_index(boxed.as_ref()) {
                scope.modules.insert(module.name.to_owned(), indexed_module);
            }
        }
        Expression::BinaryOperation(operation) => {
            if operation.operator == BinaryOperator::Match {
                let left_ids = extract_identifiers(&operation.left);
                scope.variables.extend(left_ids)
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        indexer::{FunctionIndex, ScopeIndex},
        loc,
        simple_parser::{Expression, Parser, Point},
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
                location: loc!(4, 16),
                is_private: false,
                name: "func".to_owned(),
                doc: None,
                spec: None,
                scope: Rc::new(RefCell::new(ScopeIndex {
                    parent: Some(Rc::new(RefCell::new(ScopeIndex::default()))),
                    ..ScopeIndex::default()
                })),
            }],
            ..ModuleIndex::default()
        };

        assert_eq!(result, target)
    }

    #[test]
    fn index_module_variables() {
        let code = r#"
        defmodule MyModule do
            @moduledoc """
            moduledoc
            """

            var = 10
        end
        "#;
        let result = index(code);
        let target = ModuleIndex {
            name: "MyModule".to_owned(),
            docs: Some("\n            moduledoc\n            ".to_owned()),
            scope: Rc::new(RefCell::new(ScopeIndex {
                variables: vec!["var".to_owned()],
                ..Default::default()
            })),
            ..Default::default()
        };

        assert_eq!(result, target)
    }

    // Also make tests specific to the function that extracts variable names from scopes

    // TODO: make tests only on the "function" indexer function so we avoid a lot of module boilerplate
    // leave one test to test that the module scope is the parent of the function's scope
    #[test]
    fn index_function_variables() {
        let code = r#"
        defmodule MyModule do
            @moduledoc false

            def a do
                in_func = 1
            end
        end
        "#;
        let result = index(code);

        let parent_scope = Rc::new(RefCell::new(Default::default()));

        let target = ModuleIndex {
            name: "MyModule".to_owned(),
            docs: None,
            scope: parent_scope.clone(),
            functions: vec![FunctionIndex {
                location: loc!(4, 16),
                is_private: false,
                name: "a".to_string(),
                doc: None,
                spec: None,
                scope: Rc::new(RefCell::new(ScopeIndex {
                    variables: vec!["in_func".to_owned()],
                    parent: Some(parent_scope),
                    ..Default::default()
                })),
            }],
            ..Default::default()
        };

        assert_eq!(result, target)
    }

    // #[test]
    // fn index_function_docs_and_specs() {
    //     let code = r#"
    //     defmodule MyModule do
    //         @doc "sums two numbers"
    //         @spec func(integer(), integer()) :: integer()
    //         defp func(a, b) do
    //             a + b
    //         end

    //         defp no_docs(_) do

    //         end
    //     end
    //     "#;
    //     let result = index(code);
    //     let target = ModuleIndex {
    //         name: "MyModule".to_owned(),
    //         docs: None,
    //         functions: vec![
    //             FunctionIndex {
    //                 is_private: true,
    //                 name: "func".to_owned(),
    //                 doc: Some("sums two numbers".to_string()),
    //                 spec: None,
    //             },
    //             FunctionIndex {
    //                 is_private: true,
    //                 name: "no_docs".to_owned(),
    //                 doc: None,
    //                 spec: None,
    //             },
    //         ],
    //         ..ModuleIndex::default()
    //     };

    //     assert_eq!(result, target)
    // }

    // #[test]
    // fn index_module_types() {
    //     let code = r#"
    //     defmodule MyModule do
    //         @type t :: integer()
    //         @type string :: String.t()
    //     end
    //     "#;
    //     let result = index(code);
    //     let target = ModuleIndex {
    //         name: "MyModule".to_owned(),
    //         types: vec![
    //             ("t".to_owned(), int!(10)),
    //             ("string".to_owned(), int!(10))
    //         ],
    //         ..ModuleIndex::default()
    //     };

    //     assert_eq!(result, target)
    // }
}

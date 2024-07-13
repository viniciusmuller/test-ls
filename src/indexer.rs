use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

// TODO: index nested modules

use crate::simple_parser::{BinaryOperator, Expression, Module, Point};

#[derive(Debug, PartialEq, Eq)]
pub struct Index {
    pub scopes: Vec<ScopeIndex>,
    pub module: ModuleIndex,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleIndex {
    pub location: Point,
    pub name: String,
    pub docs: Option<String>,
    pub scope_index: usize,
    pub attributes: Vec<(String, Expression)>,
    pub types: Vec<(String, Expression)>,
    // TODO: group function clauses into a single function structure by name and arity
    pub functions: Vec<FunctionIndex>,
    pub macros: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionIndex {
    pub location: Point,
    pub is_private: bool,
    pub name: String,
    pub doc: Option<String>,
    pub spec: Option<Expression>,
    pub scope_index: usize,
    // TODO: probably will need to store function body to be able to compute lambda scopes
}

impl Default for ModuleIndex {
    fn default() -> Self {
        Self {
            scope_index: 0,
            location: Point { line: 1, column: 0 },
            docs: None,
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
    parent_index: Option<usize>,
    imports: HashSet<String>,
    aliases: HashSet<String>,
    requires: HashSet<String>,
    variables: HashSet<String>,
    modules: HashMap<String, ModuleIndex>,
}

impl Default for ScopeIndex {
    fn default() -> Self {
        Self {
            parent_index: None,
            imports: HashSet::new(),
            aliases: HashSet::new(),
            requires: HashSet::new(),
            variables: HashSet::new(),
            modules: HashMap::new(),
        }
    }
}

pub fn build_index(module: &Module) -> Option<Index> {
    let mut scopes = vec![];

    let module_index = build_module_index(&module, &mut scopes);
    let index = Index {
        scopes: scopes
            .into_iter()
            .map(|s| Rc::try_unwrap(s).unwrap().into_inner())
            .collect::<Vec<_>>(),
        module: module_index,
    };
    Some(index)
}

fn build_module_index(module: &Module, scopes: &mut Vec<Rc<RefCell<ScopeIndex>>>) -> ModuleIndex {
    let mut last_doc: Option<String> = None;
    let mut last_spec: Option<Expression> = None;
    let scope = Rc::new(RefCell::new(ScopeIndex::default()));

    let mut module_index = ModuleIndex {
        location: module.location.clone(),
        name: module.name.to_owned(),
        ..ModuleIndex::default()
    };

    scopes.push(scope.clone());

    match module.body.as_ref() {
        Expression::Block(block) => {
            for expression in block {
                match expression {
                    // Expression::Module(module) => {
                    //     let boxed = Box::new(module);

                    //     if let Some(indexed_module) = build_index(boxed.as_ref()) {
                    //         module_scope
                    //             .modules
                    //             .insert(module.name.to_owned(), indexed_module);
                    //     }
                    // }
                    Expression::BinaryOperation(operation) => {
                        if operation.operator == BinaryOperator::Match {
                            let left_ids = extract_identifiers(&operation.left);
                            scope.borrow_mut().variables.extend(left_ids)
                        }
                    }
                    Expression::Alias(aliases) => {
                        scope.borrow_mut().aliases.extend(aliases.clone());
                    }
                    Expression::Require(requires) => {
                        scope.borrow_mut().requires.extend(requires.clone());
                    }
                    Expression::Import(imports) => {
                        scope.borrow_mut().imports.extend(imports.clone());
                    }
                    Expression::FunctionDef(function) => {
                        let function_scope = build_function_scope_index(&expression, Some(0));
                        scopes.push(Rc::new(RefCell::new(function_scope)));

                        module_index.functions.push(FunctionIndex {
                            location: function.location.clone(),
                            is_private: function.is_private,
                            name: function.name.to_owned(),
                            doc: last_doc.clone(),
                            spec: last_spec.clone(),
                            scope_index: scopes.len() - 1,
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
                                module_index.docs = Some(s.to_owned());
                            };
                        }
                        _ => module_index
                            .attributes
                            .push((name.to_owned(), *body.clone())),
                    },

                    // TODO: apparently the scope inside (a;b) blocks leak to the outer scope in the compiler
                    // Expression::Scope(_) => todo!(),
                    _ => {}
                }
            }
        }
        _ => {}
    }

    module_index
}

fn extract_identifiers(ast: &Expression) -> HashSet<String> {
    let mut buffer = HashSet::new();
    do_extract_identifiers(ast, &mut buffer);
    buffer
}

fn do_extract_identifiers(ast: &Expression, buffer: &mut HashSet<String>) {
    match ast {
        Expression::Tuple(expressions) => {
            for expr in expressions {
                do_extract_identifiers(&expr, buffer);
            }
        }
        Expression::FunctionDef(function) => {
            for param in &function.parameters {
                do_extract_identifiers(param, buffer);
            }

            match &*function.body {
                Expression::Block(block) => {
                    for expr in block {
                        do_extract_identifiers(expr, buffer);
                    }
                }
                _ => {}
            }

            do_extract_identifiers(&function.body, buffer);
        }
        Expression::List(list) => {
            for expr in &list.body {
                do_extract_identifiers(expr, buffer);
            }

            if let Some(ref expr) = list.cons {
                do_extract_identifiers(expr, buffer);
            }
        }
        Expression::BinaryOperation(operation) => {
            do_extract_identifiers(&operation.left, buffer);
            do_extract_identifiers(&operation.right, buffer);
        }
        Expression::Identifier(id) => {
            buffer.insert(id.to_string());
        }
        _ => {}
    }
}

fn build_function_scope_index(function: &Expression, parent_index: Option<usize>) -> ScopeIndex {
    let mut scope = ScopeIndex {
        parent_index,
        ..ScopeIndex::default()
    };

    let identifiers = extract_identifiers(function);
    scope.variables.extend(identifiers);
    parse_scope_expression(&function, &mut scope);
    scope
}

fn parse_scope_expression(expression: &Expression, scope: &mut ScopeIndex) {
    match expression {
        // Expression::Module(module) => {
        //     let boxed = Box::new(module);

        //     if let Some(indexed_module) = build_index(boxed.as_ref()) {
        //         scope.modules.insert(module.name.to_owned(), indexed_module);
        //     }
        // }
        Expression::Block(block) => {
            for expression in block {
                parse_scope_expression(expression, scope)
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::collections::HashSet;

    use crate::{
        indexer::{extract_identifiers, FunctionIndex, ScopeIndex},
        loc,
        simple_parser::{Expression, Parser, Point},
    };

    use super::{build_index, Index, ModuleIndex};

    fn index_module(code: &str) -> Index {
        if let Expression::Module(m) = parse(code) {
            build_index(&m).unwrap()
        } else {
            panic!("failed to parse module")
        }
    }

    fn parse(code: &str) -> Expression {
        let parser = Parser::new(code.to_owned(), "nofile".to_owned());
        parser.parse()
    }

    #[test]
    fn index_simple_module() {
        let code = r#"
        defmodule MyModule do
            @moduledoc "this is a nice module!"

            require Logger
            import Ecto.Query
            alias OtherModule
            alias OtherModule.{A, B}

            var = 10

            def func(a, b) do
                a + b
            end
        end
        "#;
        let result = index_module(code);
        let parent_scope = ScopeIndex {
            parent_index: None,
            imports: HashSet::from_iter(vec!["Ecto.Query".to_owned()]),
            requires: HashSet::from_iter(vec!["Logger".to_owned()]),
            aliases: HashSet::from_iter(vec![
                "OtherModule".to_owned(),
                "OtherModule.A".to_owned(),
                "OtherModule.B".to_owned(),
            ]),
            variables: HashSet::from_iter(vec!["var".to_owned()]),
            ..ScopeIndex::default()
        };
        let func_scope = ScopeIndex {
            variables: HashSet::from_iter(vec!["a".to_owned(), "b".to_owned()]),
            parent_index: Some(0),
            ..ScopeIndex::default()
        };
        let target = Index {
            scopes: vec![parent_scope, func_scope],
            module: ModuleIndex {
                location: loc!(1, 18),
                name: "MyModule".to_owned(),
                docs: Some("this is a nice module!".to_owned()),
                scope_index: 0,
                functions: vec![FunctionIndex {
                    location: loc!(11, 16),
                    is_private: false,
                    name: "func".to_owned(),
                    doc: None,
                    spec: None,
                    scope_index: 1,
                }],
                ..ModuleIndex::default()
            },
        };

        assert_eq!(result, target)
    }

    #[test]
    fn test_def_scope_gets_module_scope() {
        let code = r#"
        defmodule MyModule do
            @moduledoc false

            parent_variable = 10

            @doc "sums two numbers"
            # TODO: parse spec @spec func(integer(), integer()) :: integer()
            def a do
            end

            defp no_docs(_) do

            end
        end
        "#;
        let result = index_module(code);

        let parent_scope = ScopeIndex {
            parent_index: None,
            variables: HashSet::from_iter(vec!["parent_variable".to_owned()]),
            ..Default::default()
        };

        let a_scope = ScopeIndex {
            parent_index: Some(0),
            ..Default::default()
        };

        let no_docs_scope = ScopeIndex {
            variables: HashSet::from_iter(vec!["_".to_owned()]),
            parent_index: Some(0),
            ..Default::default()
        };

        let target = Index {
            scopes: vec![parent_scope, a_scope, no_docs_scope],
            module: ModuleIndex {
                location: loc!(1, 18),
                name: "MyModule".to_owned(),
                docs: None,
                scope_index: 0,
                functions: vec![
                    FunctionIndex {
                        location: loc!(8, 16),
                        is_private: false,
                        name: "a".to_string(),
                        doc: Some("sums two numbers".to_string()),
                        spec: None,
                        scope_index: 1,
                    },
                    FunctionIndex {
                        location: loc!(11, 17),
                        is_private: true,
                        name: "no_docs".to_string(),
                        doc: None,
                        spec: None,
                        scope_index: 2,
                    },
                ],
                ..Default::default()
            },
        };

        assert_eq!(result, target)
    }

    #[test]
    fn extract_variable_from_simple_match() {
        let code = "
        def a do
            a = 1
        end
        ";
        let func = parse(&code);
        let result = extract_identifiers(&func);
        let target = HashSet::from_iter(vec!["a".to_string()]);

        assert_eq!(result, target)
    }

    #[test]
    fn extract_variable_from_function_parameters() {
        let code = "
        def function(a, b, [1, 2, c] = b) do
        end
        ";
        let func = parse(&code);
        let result = extract_identifiers(&func);
        let target = HashSet::from_iter(vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        assert_eq!(result, target)
    }

    #[test]
    fn extract_variable_from_tuple_match() {
        let code = "
        def tuple_func do
            {:ok, result} = {:ok, 10}
        end
        ";
        let func = parse(&code);
        let result = extract_identifiers(&func);
        let target = HashSet::from_iter(vec!["result".to_string()]);

        assert_eq!(result, target)
    }

    #[test]
    fn extract_variable_from_list_match() {
        let code = "
        def list_func do
            [a, 2, 3] = [1, 2, 3]
        end
        ";
        let func = parse(&code);
        let result = extract_identifiers(&func);
        let target = HashSet::from_iter(vec!["a".to_string()]);

        assert_eq!(result, target)
    }

    #[test]
    fn extract_variable_from_list_cons_match() {
        let code = "
        def list_func do
            [1, 2 | rest] = [1, 2, 3]
        end
        ";
        let func = parse(&code);
        let result = extract_identifiers(&func);
        let target = HashSet::from_iter(vec!["rest".to_string()]);

        assert_eq!(result, target)
    }

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

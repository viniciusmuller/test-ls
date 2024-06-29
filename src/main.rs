use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use tree_sitter::{Node, Tree};

#[derive(Debug, PartialEq, Eq)]
struct Identifier(String);

#[derive(Debug, PartialEq)]
struct Float(f64);

impl Eq for Float {}

#[derive(Debug, PartialEq, Eq)]
struct Atom(String);

#[derive(Debug, PartialEq, Eq)]
struct Function {
    name: Identifier,
    is_private: bool,
    block: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
struct Module {
    name: Atom,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
struct Call {
    // TODO: document
    target: Identifier,
    remote_callee: Option<Identifier>,
    arguments: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
enum Expression {
    String(String),
    Float(Float),
    Integer(i128),
    Atom(Atom),
    Identifier(Identifier),
    Block(Vec<Expression>),
    Module(Module),
    FunctionDef(Function),
    // TODO: anonymous functions
    // TODO: function capture
    // TODO: macro
    Call(Call),
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn main() {
    let code = "
    def func do
        priv_func()
    end
    ";

    let result = parse(&code);
    dbg!(result);
}

fn parse(code: &str) -> Option<Expression> {
    let tree = get_tree(code);
    let root_node = tree.root_node();
    let mut nodes = vec![];
    flatten_node_children(code, root_node, &mut nodes);
    dbg!(&nodes);
    let tokens = nodes.clone();
    let (result, _offset) = parse_many_expressions(&code, &tokens, 0).unwrap();
    Some(Expression::Block(result))
}

fn flatten_node_children<'a>(code: &str, node: Node<'a>, vec: &mut Vec<Node<'a>>) {
    if node.child_count() > 0 {
        let mut walker = node.walk();

        for node in node.children(&mut walker) {
            vec.push(node);
            flatten_node_children(code, node, vec);
        }
    }
}

fn try_parse_module(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Expression, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    let (Identifier(identifier), offset) = try_parse_identifier(&code, tokens, offset)?;
    if identifier != "defmodule" {
        return None;
    }

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (module_name, offset) = try_parse_alias(code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)?;
    dbg!(&do_block);

    let module = Expression::Module(Module {
        name: module_name,
        body: Box::new(do_block),
    });
    Some((module, offset))
}

fn try_parse_function_definition(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Option<(Expression, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    let (offset, is_private) = match try_parse_keyword(code, tokens, offset, "defp") {
        Some((_, offset)) => (offset, true),
        None => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        match try_parse_keyword(code, tokens, offset, "def") {
            Some((_, offset)) => (offset, false),
            None => (offset, is_private),
        }
    } else {
        (offset, is_private)
    };

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (function_name, offset) = try_parse_identifier(&code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)?;

    let function = Expression::FunctionDef(Function {
        name: function_name,
        block: Box::new(do_block),
        is_private,
    });
    Some((function, offset))
}

fn try_parse_grammar_name<'a>(
    tokens: &Vec<Node<'a>>,
    offset: u64,
    target: &str,
) -> Option<(Node<'a>, u64)> {
    let token = tokens[offset as usize];

    if token.grammar_name() == target {
        Some((token, offset + 1))
    } else {
        None
    }
}

fn try_parse_identifier<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> Option<(Identifier, u64)> {
    let token = tokens[offset as usize];

    if token.grammar_name() == "identifier" {
        let identifier_name = extract_node_text(code, token);
        Some((Identifier(identifier_name), offset + 1))
    } else {
        None
    }
}

fn try_parse_keyword<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
    keyword: &str,
) -> Option<(Identifier, u64)> {
    let token = tokens[offset as usize];

    if token.grammar_name() == "identifier" {
        let identifier_name = extract_node_text(code, token);

        if identifier_name == keyword {
            Some((Identifier(identifier_name), offset + 1))
        } else {
            None
        }
    } else {
        None
    }
}

fn try_parse_call(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Expression, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    // these are optional (present in remote calls, not present for local calls)
    let (offset, has_dot) = match try_parse_grammar_name(tokens, offset, "dot") {
        Some((_node, new_offset)) => (new_offset, true),
        None => (offset, false),
    };

    let offset = match try_parse_grammar_name(tokens, offset, ".") {
        Some((_node, new_offset)) => new_offset,
        None => offset,
    };

    let mut remote_callee = None;

    let offset = match try_parse_grammar_name(tokens, offset, "alias") {
        Some((node, new_offset)) => {
            remote_callee = Some(Identifier(extract_node_text(code, node)));
            new_offset
        }
        None => offset,
    };

    let offset = if remote_callee.is_none() && has_dot {
        let (token, new_offset) = try_parse_grammar_name(tokens, offset, "identifier")?;
        remote_callee = Some(Identifier(extract_node_text(code, token)));
        new_offset
    } else {
        offset
    };

    let offset = match try_parse_grammar_name(tokens, offset, ".") {
        Some((_node, new_offset)) => new_offset,
        None => offset,
    };

    let (identifier, offset) = try_parse_identifier(code, tokens, offset)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    // TODO: if we match a opening, we need to match a closing one as well
    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Some((_node, new_offset)) => (new_offset, true),
        None => (offset, false),
    };

    let (arguments, offset) = match parse_expressions_sep_by_comma(code, tokens, offset) {
        None => (vec![], offset),
        Some((expressions, new_offset)) => (expressions, new_offset),
    };

    let offset = if has_parenthesis {
        let (_node, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    Some((
        Expression::Call(Call {
            target: identifier,
            remote_callee,
            arguments,
        }),
        offset,
    ))
}

// TODO: use this to parse expressions sep by comma in the function body
fn parse_expressions_sep_by_comma(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Option<(Vec<Expression>, u64)> {
    let (expr, offset) = try_parse_expression(code, tokens, offset)?;
    let mut offset_mut = offset;
    let mut expressions = vec![expr];

    // TODO: Having to add these checks feel weird
    if offset == tokens.len() as u64 {
        return Some((expressions, offset));
    }

    let mut next_comma = try_parse_grammar_name(tokens, offset, ",");

    while next_comma.is_some() {
        let (_next_comma, offset) = next_comma.unwrap();
        let (expr, offset) = try_parse_expression(code, tokens, offset)?;

        expressions.push(expr);

        if offset == tokens.len() as u64 {
            return Some((expressions, offset));
        }

        offset_mut = offset;
        next_comma = try_parse_grammar_name(tokens, offset, ",");
    }

    Some((expressions, offset_mut))
}

fn try_parse_do_block<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> Option<(Expression, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do_block")?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "do")?;

    let (body, offset) = parse_many_expressions(code, tokens, offset)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "end")?;

    Some((Expression::Block(body), offset))
}

fn try_parse_atom(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Atom, u64)> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "atom")?;
    let atom_string = extract_node_text(code, atom_node);
    Some((Atom(atom_string), offset))
}

fn try_parse_integer(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(i128, u64)> {
    let (integer_node, offset) = try_parse_grammar_name(tokens, offset, "integer")?;
    let integer_text = extract_node_text(code, integer_node);
    let integer_text = integer_text.replace("_", "");

    match integer_text.parse::<i128>() {
        Ok(n) => Some((n, offset)),
        Err(_reason) => {
            println!("Failed to parse integer: {}", integer_text);
            None
        }
    }
}

fn try_parse_float(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Float, u64)> {
    let (float_node, offset) = try_parse_grammar_name(tokens, offset, "float")?;
    let float_text = extract_node_text(code, float_node);

    match float_text.parse::<f64>() {
        Ok(n) => Some((Float(n), offset)),
        Err(_reason) => {
            println!("Failed to parse float: {}", float_text);
            None
        }
    }
}

fn try_parse_string(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(String, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "string")?;
    let quotes = vec!["\"", "\"\"\""];
    let (_, offset) = try_parse_either_grammar_name(tokens, offset, &quotes)?;
    let (string_node, offset) = try_parse_grammar_name(tokens, offset, "quoted_content")?;
    let string_text = extract_node_text(code, string_node);
    let (_, offset) = try_parse_either_grammar_name(tokens, offset, &quotes)?;
    Some((string_text, offset))
}

fn try_parse_either_grammar_name<'a>(
    tokens: &Vec<Node<'a>>,
    offset: u64,
    grammar_names: &Vec<&str>,
) -> Option<(Node<'a>, u64)> {
    for grammar_name in grammar_names {
        match try_parse_grammar_name(tokens, offset, grammar_name) {
            Some(result) => return Some(result),
            None => {}
        }
    }

    None
}

/// Grammar_name = alias means module names in the TS elixir grammar.
/// e.g: ThisIsAnAlias, Elixir.ThisIsAlsoAnAlias
/// as these are also technically atoms, we return them as atoms
fn try_parse_alias(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Atom, u64)> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "alias")?;
    let atom_string = extract_node_text(code, atom_node);
    Some((Atom(atom_string), offset))
}

fn try_parse_expression<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> Option<(Expression, u64)> {
    try_parse_module(code, tokens, offset)
        .or_else(|| try_parse_function_definition(code, tokens, offset))
        .or_else(|| try_parse_call(code, tokens, offset))
        .or_else(|| try_parse_do_block(code, tokens, offset))
        .or_else(|| {
            try_parse_alias(code, tokens, offset)
                .and_then(|(alias, offset)| Some((Expression::Atom(alias), offset)))
        })
        .or_else(|| {
            try_parse_atom(code, tokens, offset)
                .and_then(|(atom, offset)| Some((Expression::Atom(atom), offset)))
        })
        .or_else(|| {
            try_parse_identifier(code, tokens, offset)
                .and_then(|(identifier, offset)| Some((Expression::Identifier(identifier), offset)))
        })
        .or_else(|| {
            try_parse_integer(code, tokens, offset)
                .and_then(|(integer, offset)| Some((Expression::Integer(integer), offset)))
        })
        .or_else(|| {
            try_parse_float(code, tokens, offset)
                .and_then(|(float, offset)| Some((Expression::Float(float), offset)))
        })
        .or_else(|| {
            try_parse_string(code, tokens, offset)
                .and_then(|(string, offset)| Some((Expression::String(string), offset)))
        })
}

fn parse_many_expressions(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Option<(Vec<Expression>, u64)> {
    let (expression, offset) = match try_parse_expression(code, tokens, offset) {
        Some(result) => result,
        None => {
            return Some((vec![], offset));
        }
    };
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    if offset == tokens.len() as u64 {
        return Some((expressions, offset_mut));
    }

    loop {
        match try_parse_expression(code, tokens, offset_mut) {
            Some((expression, offset)) => {
                expressions.push(expression);
                offset_mut = offset;

                if offset == tokens.len() as u64 {
                    return Some((expressions, offset_mut));
                }
            }
            None => {
                return Some((expressions, offset_mut));
            }
        }
    }
}

fn extract_node_text(code: &str, node: Node) -> String {
    code[node.byte_range()].to_string()
}

#[cfg(test)]
mod tests {
    use super::Expression::Block;
    use super::*;

    #[test]
    fn parse_atom() {
        let code = ":atom";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Atom(Atom(":atom".to_string()))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_alias() {
        let code = "Elixir.MyModule";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Atom(Atom("Elixir.MyModule".to_string()))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_identifier() {
        let code = "my_variable_name";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Identifier(Identifier(
            "my_variable_name".to_string(),
        ))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_remote_call_module() {
        let code = "Enum.reject(a, my_func)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("reject".to_string()),
            remote_callee: Some(Identifier("Enum".to_string())),
            arguments: vec![
                Expression::Identifier(Identifier("a".to_string())),
                Expression::Identifier(Identifier("my_func".to_string())),
            ],
        })]);

        assert_eq!(result, target);
    }

    // TODO: module functions vs lambdas

    #[test]
    fn parse_remote_call_identifier() {
        let code = "json_parser.parse(body)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("parse".to_string()),
            remote_callee: Some(Identifier("json_parser".to_string())),
            arguments: vec![Expression::Identifier(Identifier("body".to_string()))],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_local_call() {
        let code = "to_integer(a)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("to_integer".to_string()),
            remote_callee: None,
            arguments: vec![Expression::Identifier(Identifier("a".to_string()))],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_parenthesis() {
        let code = "to_integer a";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("to_integer".to_string()),
            remote_callee: None,
            arguments: vec![Expression::Identifier(Identifier("a".to_string()))],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_parenthesis_multiple_args() {
        let code = "to_integer a, 10, 30.2";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("to_integer".to_string()),
            remote_callee: None,
            arguments: vec![
                Expression::Identifier(Identifier("a".to_string())),
                Expression::Integer(10),
                Expression::Float(Float(30.2)),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_arguments() {
        let code = "to_integer()";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Identifier("to_integer".to_string()),
            remote_callee: None,
            arguments: vec![],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_float() {
        let code = "34.39212";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Float(Float(34.39212))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_integer() {
        let code = "1000";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Integer(1000)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_integer_with_underscores() {
        let code = "100_000";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Integer(100_000)]);

        assert_eq!(result, target);
    }

    // TODO: parse integer in different bases
    // TODO: parse integer in scientific notation

    #[test]
    fn parse_string() {
        let code = "\"string!\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::String("string!".to_string())]);

        assert_eq!(result, target);
    }

    // TODO: support string interpolation inside of quoted_content
    // #[test]
    // fn parse_string_interpolation() {
    //     let code = "\"this is #{string_value}!\"";
    //     let result = parse(&code).unwrap();

    //     let target = Block(vec![Expression::String("\"string!\"".to_string())]);

    //     assert_eq!(result, target);
    // }

    #[test]
    fn parse_multiline_string() {
        let code = "
        \"\"\"
        Multiline!
        String!
        \"\"\"
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::String(
            "\n        Multiline!\n        String!\n        ".to_string(),
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_simple_module_definition() {
        let code = "
        defmodule Test.CustomModule do

        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Module(Module {
            name: Atom("Test.CustomModule".to_string()),
            body: Box::new(Expression::Block(vec![])),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_multiple_modules_definition() {
        let code = "
        defmodule Test.CustomModule do

        end

        defmodule Test.SecondModule do

        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![
            Expression::Module(Module {
                name: Atom("Test.CustomModule".to_string()),
                body: Box::new(Expression::Block(vec![])),
            }),
            Expression::Module(Module {
                name: Atom("Test.SecondModule".to_string()),
                body: Box::new(Expression::Block(vec![])),
            }),
        ]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_module_functions() {
        let code = "
        defmodule Test.CustomModule do
            def func do
                priv_func()
            end

            defp priv_func do
                10
            end
        end
        ";
        let result = parse(&code).unwrap();

        // TODO: try to reduce boilerplate in assertions
        let target = Block(vec![Expression::Module(Module {
            name: Atom("Test.CustomModule".to_string()),
            body: Box::new(Block(vec![
                Expression::FunctionDef(Function {
                    name: Identifier("func".to_string()),
                    is_private: false,
                    block: Box::new(Block(vec![Expression::Call(Call {
                        target: Identifier("priv_func".to_string()),
                        remote_callee: None,
                        arguments: vec![],
                    })])),
                }),
                Expression::FunctionDef(Function {
                    name: Identifier("priv_func".to_string()),
                    is_private: true,
                    block: Box::new(Block(vec![Expression::Integer(10)])),
                }),
            ])),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_definition() {
        let code = "
        def func do
            priv_func()
        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            block: Box::new(Block(vec![Expression::Call(Call {
                target: Identifier("priv_func".to_string()),
                remote_callee: None,
                arguments: vec![],
            })])),
        })]);

        assert_eq!(result, target);
    }
}

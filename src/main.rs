use std::{collections::VecDeque, sync::{Arc, Mutex}};

use tree_sitter::{Node, Tree};

#[derive(Debug)]
struct Identifier(String);

#[derive(Debug)]
struct Atom(String);

#[derive(Debug)]
struct Function {
    name: Identifier,
    is_private: bool
}

#[derive(Debug)]
struct Module {
    name: Atom,
    functions: Vec<Function>
}

#[derive(Debug)]
struct Call {
    // TODO: document
    target: Identifier,
    remote_callee: Option<Identifier>,
    arguments: Vec<Expression>
}

#[derive(Debug)]
enum Expression {
    Atom(Atom),
    Identifier(Identifier),
    Call(Call),
    Block(Vec<Expression>),
    Module(Module),
    Function(Function),
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_elixir::language()).expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn main() {
    let code = "
    defmodule MyModule.Name do
        def b do
            10
        end
    end

    defmodule MySecondModule.Name do
        def b do
            10
        end

        # TODO: test one-line form of def and do blocks and see if it works
    end
    ";

    // TODO: start test suite
    let code = "
    Enum.parse(a, b)
    ";

    let code = "
    to_integer(a, b)
    ";

    let code = "
    module.to_integer(a, b)
    ";

    let code = "
    defmodule MySecondModule.Name do

    end
    ";

    let code = "
    :test
    Test
    ";
    let tree = get_tree(code);

    let root_node = tree.root_node();
    let mut nodes = vec![];
    print_node_children(code, root_node, &mut nodes);

    // TODO: drop comments
    let tokens = nodes.clone();
    dbg!(&tokens);

    // let (module, offset) = try_parse_module(&code, &tokens, 0).unwrap();
    // let (result, offset) = try_parse_expression(&code, &tokens, 0).unwrap();

    let (result, offset) = parse_many_expressions(&code, &tokens, 0).unwrap();
    dbg!(result);

    // TODO: offset should correctly be returned at the end of the module block (parse the rest of the module block)
    // let (module_2, _offset) = try_parse_module(&code, &tokens, offset).unwrap();
    // dbg!(module);
}

fn print_node_children<'a>(code: &str, node: Node<'a>, vec: &mut Vec<Node<'a>>) {
    if node.child_count() > 0 {
        let mut walker = node.walk();

        for node in node.children(&mut walker) {
            vec.push(node);
            print_node_children(code, node, vec);
        }
    }
}

fn try_parse_module(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Expression, u64)> {
    let _ = try_parse_grammar_name(tokens, offset, "call")?;

    let (_, offset) = try_parse_identifier(&code, tokens, offset, "defmodule")?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (module_name, offset) = try_parse_alias(code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)?;
    dbg!(do_block);

    let module = Expression::Module(Module{name: module_name, functions: vec![]});
    Some((module, offset))
}

fn try_parse_grammar_name<'a>(tokens: &Vec<Node<'a>>, offset: u64, target: &str) -> Option<(Node<'a>, u64)> {
    let token = tokens[offset as usize];

    if token.grammar_name() == target {
        Some((token, offset + 1))
    } else {
        None
    }
}

fn try_parse_identifier<'a>(code: &str, tokens: &Vec<Node<'a>>, offset: u64, target: &str) -> Option<(Node<'a>, u64)> {
    let token = tokens[offset as usize];

    if token.grammar_name() == "identifier" && &code[token.byte_range()] == target {
        Some((token, offset + 1))
    } else {
        None
    }
}

fn try_parse_call(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Expression, u64)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "dot")?;

    // module.to_integer(a, b)
    // alias can also be an identifier in case the module was bound to a variable
    // vvvv

    // these are optional (present in remote calls, not present for local calls)
    let (_, offset) = try_parse_grammar_name(tokens, offset, "alias")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, ".")?;

    let (node, offset) = try_parse_grammar_name(tokens, offset, "identifier")?;
    let callee_name = extract_node_text(code, node);
    dbg!(callee_name);

    let _ = try_parse_grammar_name(tokens, offset, "arguments")?;

    todo!();
}

fn try_parse_do_block<'a>(code: &str, tokens: &Vec<Node<'a>>, offset: u64) -> Option<(Expression, u64)> {
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

/// Grammar_name = alias means module names in the TS elixir grammar.
/// e.g: ThisIsAnAlias, Elixir.ThisIsAlsoAnAlias
/// as these are also technically atoms, we return them as atoms
fn try_parse_alias(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Atom, u64)> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "alias")?;
    let atom_string = extract_node_text(code, atom_node);
    Some((Atom(atom_string), offset))
}

fn try_parse_expression<'a>(code: &str, tokens: &Vec<Node<'a>>, offset: u64) -> Option<(Expression, u64)> {
    try_parse_module(code, tokens, offset)
        .or_else(|| try_parse_do_block(code, tokens, offset))
        .or_else(|| try_parse_call(code, tokens, offset))
        .or_else(|| try_parse_alias(code, tokens, offset)
                .and_then(|(alias, offset)| Some((Expression::Atom(alias), offset))))
        .or_else(|| try_parse_atom(code, tokens, offset)
                .and_then(|(atom, offset)| Some((Expression::Atom(atom), offset))))
}

fn parse_many_expressions(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Vec<Expression>, u64)> {
    let (expression, offset) = try_parse_expression(code, tokens, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    loop {
        match try_parse_expression(code, tokens, offset_mut) {
            Some((expression, offset)) => {
                expressions.push(expression);
                offset_mut = offset;

                if offset == expressions.len() as u64 {
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

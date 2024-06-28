use std::{collections::VecDeque, sync::{Arc, Mutex}};

use tree_sitter::{Node, Tree};

#[derive(Debug)]
struct Function {
    name: String,
    is_private: bool
}

#[derive(Debug)]
struct Module {
    name: String,
    functions: Vec<Function>
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
    let tree = get_tree(code);

    let root_node = tree.root_node();
    let mut nodes = vec![];
    print_node_children(code, root_node, &mut nodes);

    let tokens = nodes.clone();
    dbg!(&tokens);

    let (module, offset) = try_parse_module(&code, &tokens, 0).unwrap();

    // TODO: offset should correctly be returned at the end of the module block (parse the rest of the module block)
    let (module_2, _offset) = try_parse_module(&code, &tokens, offset).unwrap();
    dbg!(module);
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

fn try_parse_module(code: &str, tokens: &Vec<Node>, offset: u64) -> Option<(Module, u64)> {
    // TODO: develop underlying abstraction to automated these "bind operations"
    // and offset tracking

    let mut offset_mut = offset;
    let _ = try_parse_grammar_name(tokens[offset_mut as usize], "call")?;
    offset_mut += 1;

    let _ = try_parse_identifier(&code, tokens[offset_mut as usize], "defmodule")?;
    offset_mut += 1;

    let _ = try_parse_grammar_name(tokens[offset_mut as usize], "arguments")?;
    offset_mut += 1;

    let module_name_node = try_parse_grammar_name(tokens[offset_mut as usize], "alias")?;
    let module_name = &code[module_name_node.byte_range()];
    offset_mut += 1;

    let _ = try_parse_grammar_name(tokens[offset_mut as usize], "do_block")?;
    offset_mut += 1;

    let _ = try_parse_grammar_name(tokens[offset_mut as usize], "do")?;
    offset_mut += 1;

    // TODO: parse until the end (and handle nested do blocks)

    dbg!(module_name);

    let result = (Module{name: module_name.to_string(), functions: vec![]}, offset_mut);
    Some(result)
}

fn try_parse_grammar_name<'a>(token: Node<'a>, target: &str) -> Option<Node<'a>> {
    if token.grammar_name() == target {
        Some(token)
    } else {
        None
    }
}

fn try_parse_identifier<'a>(code: &str, token: Node<'a>, target: &str) -> Option<Node<'a>> {
    if token.grammar_name() == "identifier" && &code[token.byte_range()] == target {
        Some(token)
    } else {
        None
    }
}

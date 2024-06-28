use tree_sitter::{Node, Tree};

fn get_tree() -> Tree {
    let code = "
    defmodule MyModule.Name do
        def b do
            10
        end
    end
    ";
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_elixir::language()).expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn main() {
    let tree = get_tree();

    let root_node = tree.root_node();
    print_node_children(root_node, 0);
}

fn print_node_children(node: Node, depth: u64) {
    let code = "
    defmodule MyModule.Name do
        def b do
            10
        end
    end
    ";

    if node.child_count() > 0 {
        let mut walker = node.walk();

        // let mut buffer = vec![0_u8; 256];
        // node.utf8_text(&mut buffer).unwrap();
        // dbg!(buffer);

        for node in node.children(&mut walker) {
            dbg!(node, depth, dbg!(&code[node.byte_range()]));
            print_node_children(node, depth + 1);
        }
    }
}

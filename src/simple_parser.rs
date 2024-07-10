use tree_sitter::{Node, Tree};
use crate::parser::Point;

#[derive(Debug)]
pub struct TSError {
    path: String,
    start: Point,
    end: Point,
}

#[derive(Debug)]
pub enum NodeExpr {
    Leaf(String),
    Node(Vec<NodeExpr>),
}

#[derive(Debug)]
pub struct TreeTraversal {
    pub has_errors: bool,
    pub ast: NodeExpr,
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn traverse_tree(root_node: &Node, path: &str) -> TreeTraversal {
    let ast = parse_node(&root_node, &path);
    let has_errors = check_tree_has_errors(&ast);

    TreeTraversal {
        has_errors,
        ast,
    }
}

fn check_tree_has_errors(node: &NodeExpr) -> bool {
    match node {
        NodeExpr::Leaf(grammar_name) => grammar_name == "ERROR",
        NodeExpr::Node(children) => children.iter().any(check_tree_has_errors)
    }
}

fn parse_node(node: &Node, path: &str) -> NodeExpr {
    let grammar_name = node.grammar_name();

    if node.child_count() == 0 {
        NodeExpr::Leaf(grammar_name.to_owned())
    } else {
        let mut cursor = node.walk();
        let mut tokens = vec![];

        for node in node.children(&mut cursor) {
            tokens.push(parse_node(&node, path));
        }

        NodeExpr::Node(tokens)
    }
}

pub fn parse(code: &str, path: &str) -> TreeTraversal {
    let tree = get_tree(code);
    let root_node = tree.root_node();
    traverse_tree(&root_node, path)
    // let mut nodes = vec![];
    // flatten_node_children(root_node, &mut nodes);

    // #[cfg(test)]
    // dbg!(&nodes);

    // let tokens = nodes.clone();

    // let mut parser = PState::init(code, tokens);
    // let (result, _offset) = parser.parse()?;

    // Ok(Expression::Block(vec![]))
}

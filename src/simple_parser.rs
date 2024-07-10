use std::fmt::Display;

use tree_sitter::{Node, Tree};

#[derive(Debug, PartialEq, Eq)]
pub struct TSError {
    path: String,
    start: Point,
    end: Point,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Point {
    line: usize,
    column: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    name: String,
    body: Box<ASTNode>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Expression {
    Module(Module),
    Unparsed(String),
    TreeSitterError(Point, Point),
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Expression::String(string) => write!(f, "String({})", string),
            Expression::Module(module) => write!(f, "Module({})", module.name),
            Expression::Unparsed(grammar_name) => write!(f, "Unparsed({})", grammar_name),
            Expression::TreeSitterError(start_pos, _) => write!(
                f,
                "TreeSitterError({}, {})",
                start_pos.line, start_pos.column
            ),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ASTNode {
    Leaf(Expression),
    Node(Vec<ASTNode>),
}

#[derive(Debug)]
pub struct TreeTraversal {
    pub has_errors: bool,
    pub ast: ASTNode,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    code: String,
    filepath: String,
}

impl Parser {
    pub fn new(code: String, filepath: String) -> Self {
        Parser { code, filepath }
    }

    pub fn parse(&self) -> TreeTraversal {
        let tree = get_tree(&self.code);
        let root_node = tree.root_node();
        let tree = self.traverse_tree(&root_node);
        // print_ast(&tree.ast, 0);
        tree
    }

    fn traverse_tree(&self, root_node: &Node) -> TreeTraversal {
        let ast = parse_node(self, &root_node);
        let has_errors = check_tree_has_errors(&ast);

        TreeTraversal { has_errors, ast }
    }

    pub fn get_text(&self, node: &Node) -> String {
        let range = node.range();
        self.code[range.start_byte..range.end_byte].to_owned()
    }
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn check_tree_has_errors(node: &ASTNode) -> bool {
    match node {
        ASTNode::Leaf(Expression::TreeSitterError(_, _)) => true,
        ASTNode::Leaf(_) => false,
        ASTNode::Node(children) => children.iter().any(check_tree_has_errors),
    }
}

fn parse_node(parser: &Parser, node: &Node) -> ASTNode {
    if node.child_count() == 0 {
        // TODO: we need to call this not only for leaf nodes
        let node = parse_expression(parser, node);
        ASTNode::Leaf(node)
    } else {
        let mut cursor = node.walk();
        let mut tokens = vec![];

        for node in node.children(&mut cursor) {
            tokens.push(parse_node(parser, &node));
        }

        ASTNode::Node(tokens)
    }
}

fn parse_expression(parser: &Parser, node: &Node) -> Expression {
    match node.grammar_name() {
        "ERROR" => build_treesitter_error_node(node),
        "identifier" => parse_identifier_node(parser, node),
        name => Expression::Unparsed(name.to_owned()),
    }
}

fn parse_identifier_node(parser: &Parser, node: &Node) -> Expression {
    let identifier_name = parser.get_text(node);
    if identifier_name == "defmodule" {
        let module = Module {
            name: identifier_name.to_owned(),
            body: Box::new(ASTNode::Leaf(Expression::Unparsed(
                "module body here!".to_owned(),
            ))),
        };
        Expression::Module(module)
    } else {
        Expression::Unparsed(node.grammar_name().to_owned())
    }
}

fn build_treesitter_error_node(node: &Node) -> Expression {
    Expression::TreeSitterError(
        tree_sitter_location_to_point(node.start_position()),
        tree_sitter_location_to_point(node.end_position()),
    )
}

fn tree_sitter_location_to_point(ts_point: tree_sitter::Point) -> Point {
    Point {
        line: ts_point.row,
        column: ts_point.column,
    }
}

fn print_ast(ast: &ASTNode, depth: usize) {
    match ast {
        ASTNode::Leaf(expr) => println!("{}└ {}", "  ".repeat(depth), expr),
        ASTNode::Node(children) => {
            for child in children {
                print_ast(child, depth + 1)
            }
        }
    }
}

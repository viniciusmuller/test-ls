use std::fmt::Display;

use tree_sitter::{Node, Tree};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TSError {
    path: String,
    start: Point,
    end: Point,
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Point {
    line: usize,
    column: usize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    name: String,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Module(Module),
    Block(Vec<Expression>),
    Call,
    // FunctionDefinition(Module),
    Identifier(String),
    Unparsed(String),
    TreeSitterError(Point, Point),
    Parent(Box<Expression>, Vec<Expression>),
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Module(module) => write!(f, "Module({})", module.name),
            Expression::Identifier(id) => write!(f, "Identifier({})", id),
            Expression::Call => write!(f, "Call()"),
            Expression::Unparsed(grammar_name) => write!(f, "Unparsed({})", grammar_name),
            Expression::Block(children) => write!(
                f,
                "Block({})",
                children
                    .iter()
                    .map(|c| format!("{}", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Expression::Parent(node, children) => write!(
                f,
                "Parent({})({})",
                node,
                children
                    .iter()
                    .map(|c| format!("{}", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Expression::TreeSitterError(start_pos, _) => write!(
                f,
                "TreeSitterError({}, {})",
                start_pos.line, start_pos.column
            ),
        }
    }
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

    pub fn parse(&self) -> Expression {
        let tree = get_tree(&self.code);
        let root_node = tree.root_node();
        let ast = match parse_node(self, &root_node) {
            // Ignore "source" root node and normalize block if single expression
            Expression::Parent(_, mut children) => normalize_block(&mut children),
            Expression::Unparsed(name) if name == "source" => Expression::Block(vec![]),
            // TODO: don't panic, just log if it ever happens
            block => {
                dbg!(&block);
                panic!("should always receive source block from tree-sitter")
            }
        };
        print_ast(&ast, 0);
        ast
    }

    pub fn get_text(&self, node: &Node) -> String {
        let range = node.range();
        self.code[range.start_byte..range.end_byte].to_owned()
    }
}

fn normalize_block(block: &mut Vec<Expression>) -> Expression {
    if block.len() == 1 {
        block.pop().unwrap()
    } else {
        Expression::Block(block.to_owned())
    }
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn parse_node(parser: &Parser, node: &Node) -> Expression {
    if node.child_count() == 0 {
        parse_expression(parser, node)
    } else {
        let mut cursor = node.walk();
        let mut tokens = vec![];

        for node in node.children(&mut cursor) {
            tokens.push(parse_node(parser, &node));
        }

        let node = parse_expression(parser, node);
        Expression::Parent(Box::new(node), tokens)
    }
}

fn parse_expression(parser: &Parser, node: &Node) -> Expression {
    let grammar_name = node.grammar_name();

    match grammar_name {
        "ERROR" => build_treesitter_error_node(node),
        "identifier" => parse_identifier_node(parser, node),
        "call" => try_parse_call_node(parser, node)
            .unwrap_or(Expression::Unparsed(grammar_name.to_owned())),
        name => Expression::Unparsed(name.to_owned()),
    }
}

fn parse_identifier_node(parser: &Parser, node: &Node) -> Expression {
    let text = parser.get_text(node);
    Expression::Identifier(text.to_owned())
}

fn try_parse_call_node(parser: &Parser, node: &Node) -> Option<Expression> {
    try_parse_module(parser, node)
}

fn try_parse_module(parser: &Parser, node: &Node) -> Option<Expression> {
    // TODO: update the logic to make this function consume all of its downstream tokens and not
    // duplicate them (ie it should return its children as well)
    let child = node.child(0).unwrap();
    let _ = try_parse_specific_identifier(parser, &child, "defmodule")?;
    todo!();
    // now there should be `arguments` with the module name as an alias
    // and then the module body as a do block/keyword syntax
}

fn try_parse_specific_identifier<'a>(
    parser: &Parser,
    node: &'a Node,
    target: &str,
) -> Option<&'a Node<'a>> {
    if parser.get_text(node) == target {
        Some(node)
    } else {
        None
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

fn print_ast(ast: &Expression, depth: usize) {
    match ast {
        Expression::Parent(expr, children) => {
            println!("{}{}└ {}", "  ".repeat(depth), depth, expr);

            for child in children {
                print_ast(child, depth + 1)
            }
        }
        expr => println!("{}{}└ {}", "  ".repeat(depth), depth, expr),
    }
}

#[cfg(test)]
mod tests {
    use super::Expression;
    use super::Parser;

    fn parse(code: &str) -> Expression {
        let parser = Parser::new(code.to_owned(), "nofile".to_owned());
        parser.parse()
    }

    #[macro_export]
    macro_rules! id {
        ($x:expr) => {
            Expression::Identifier($x.to_string())
        };
    }

    #[macro_export]
    macro_rules! unparsed {
        ($x:expr) => {
            Expression::Unparsed($x.to_string())
        };
    }

    #[macro_export]
    macro_rules! parent {
        ($x:expr, $children:expr) => {
            Expression::Parent(Box::new($x), $children)
        };
    }

    #[test]
    fn parse_identifier() {
        let code = "my_variable_name";
        let result = parse(code);
        let target = id!("my_variable_name");
        assert_eq!(result, target);
    }

    #[test]
    fn parse_tree_sitter_error() {
        let code = "<%= a %>";
        let result = parse(code);

        match result {
            Expression::Parent(p, _) => match *p {
                Expression::TreeSitterError(_, _) => {}
                _ => panic!("failed to match parent"),
            },
            _ => panic!("failed to match parent"),
        }
    }

    #[test]
    fn parse_empty_file() {
        let code = "";
        let result = parse(code);
        let expected = Expression::Block(vec![]);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_module() {
        let code = "
        defmodule MyModule do
            def a do
                10
            end
        end
        ";
        let result = parse(code);
        let expected = Expression::Block(vec![]);
        assert_eq!(result, expected)
    }
}

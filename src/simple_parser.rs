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
    pub name: String,
    pub body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub is_private: bool,
    pub parameters: Vec<Expression>,
    pub body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Scope {
    pub body: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Module(Module),
    FunctionDef(FunctionDef),
    Attribute(String, Box<Expression>),
    String(String),
    Scope(Scope),
    Identifier(String),
    Unparsed(String, Vec<Expression>),
    TreeSitterError(Point, Point),
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::FunctionDef(function) => write!(f, "FunctionDef({})", function.name),
            Expression::Module(module) => write!(f, "Module({})", module.name),
            Expression::Attribute(name, _) => write!(f, "Module({})", name),
            Expression::Scope(_) => write!(f, "Scope()"),
            Expression::String(s) => write!(f, "String({})", s),
            Expression::Identifier(id) => write!(f, "Identifier({})", id),
            Expression::Unparsed(grammar_name, _) => write!(f, "Unparsed({})", grammar_name),
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
        let ast = match parse_expression(self, &root_node) {
            // Ignore "source" root node and normalize block if single expression
            Expression::Unparsed(name, mut children) if name == "source" => {
                normalize_block(&mut children)
            }
            expr => expr,
        };
        // print_expression(&ast, 0);
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
        let scope = Scope {
            body: block.to_owned(),
        };
        Expression::Scope(scope)
    }
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn parse_expression(parser: &Parser, node: &Node) -> Expression {
    let grammar_name = node.grammar_name();

    match grammar_name {
        "ERROR" => build_treesitter_error_node(node),
        "identifier" => Expression::Identifier(parse_identifier_node(parser, node)),
        "call" => parse_call_node(parser, node),
        "string" => parse_string_node(parser, node),
        "unary_operator" => parse_unary_operator(parser, node),
        "block" => parse_block_node(parser, node),
        _unknown => build_unparsed_node(parser, node),
    }
}

fn parse_string_node(parser: &Parser, node: &Node) -> Expression {
    match try_parse_grammar_name(node, "string") {
        Some(_) => {
            let mut cursor = node.walk();
            let children = node.children(&mut cursor);

            let a = children
                .into_iter()
                .filter(|n| !["\"", "\"\"\""].contains(&n.grammar_name()))
                .map(|n| parser.get_text(&n))
                .collect::<Vec<_>>()
                .join("");

            Expression::String(a)
        }
        None => build_unparsed_node(parser, node),
    }
}

fn parse_unary_operator(parser: &Parser, node: &Node) -> Expression {
    try_parse_attribute(parser, node).unwrap_or_else(|| build_unparsed_node(parser, node))
}

fn parse_identifier_node(parser: &Parser, node: &Node) -> String {
    parser.get_text(node).to_owned()
}

fn parse_call_node(parser: &Parser, node: &Node) -> Expression {
    try_parse_module(parser, node)
        .or_else(|| try_parse_function_definition(parser, node))
        .unwrap_or_else(|| build_unparsed_node(parser, node))
}

fn parse_block_node(parser: &Parser, node: &Node) -> Expression {
    let mut result = Vec::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let grammar_name = child.grammar_name();

        if [";", "(", ")"].contains(&grammar_name) {
            continue;
        }

        result.push(parse_expression(parser, &child))
    }

    normalize_block(&mut result)
}

fn build_unparsed_node(parser: &Parser, node: &Node) -> Expression {
    let grammar_name = node.grammar_name();

    let mut cursor = node.walk();
    let mut tokens = vec![];

    for node in node.children(&mut cursor) {
        tokens.push(parse_expression(parser, &node));
    }

    Expression::Unparsed(grammar_name.to_string(), tokens)
}

fn try_parse_attribute(parser: &Parser, node: &Node) -> Option<Expression> {
    let _ = try_parse_grammar_name(node, "unary_operator")?;
    let child = node.child(0)?;
    let _ = try_parse_grammar_name(&child, "@")?;

    let call_node = child.next_sibling()?;
    let _ = try_parse_grammar_name(&call_node, "call")?;

    let attribute_node = call_node.child(0)?;
    let attribute_name = parse_identifier_node(parser, &attribute_node);

    let arguments_node = attribute_node.next_sibling()?;
    let _ = try_parse_grammar_name(&arguments_node, "arguments")?;

    let body_node = arguments_node.child(0)?;
    let body = parse_expression(parser, &body_node);
    Some(Expression::Attribute(attribute_name, Box::new(body)))
}

fn try_parse_module(parser: &Parser, node: &Node) -> Option<Expression> {
    let child = node.child(0)?;
    let _ = try_parse_specific_identifier(parser, &child, "defmodule")?;
    let child = child.next_sibling()?;
    let module_name = try_parse_module_def_name(parser, &child)?;
    let child = child.next_sibling()?;
    let body = try_parse_do_block(parser, &child)?;
    let module = Module {
        name: module_name.to_string(),
        body: Box::new(body),
    };
    Some(Expression::Module(module))
}

fn try_parse_module_def_name(parser: &Parser, node: &Node) -> Option<String> {
    let _ = try_parse_grammar_name(node, "arguments")?;
    let child = node.child(0)?;
    let alias = try_parse_alias(parser, &child);
    alias
}

fn try_parse_function_definition(parser: &Parser, node: &Node) -> Option<Expression> {
    let child = node.child(0)?;
    let caller = try_parse_either_identifier(parser, &child, &["def", "defp"])?;
    let is_private = caller == "defp";
    let def_arguments_node = child.next_sibling()?;
    let (function_name, parameters) = try_parse_function_def(parser, &def_arguments_node)?;
    let body_node = def_arguments_node.next_sibling()?;
    let body = try_parse_do_block(parser, &body_node)?;
    let function = FunctionDef {
        is_private,
        parameters,
        name: function_name.to_string(),
        body: Box::new(body),
    };
    Some(Expression::FunctionDef(function))
}

fn try_parse_function_def(parser: &Parser, node: &Node) -> Option<(String, Vec<Expression>)> {
    let _ = try_parse_grammar_name(node, "arguments")?;
    let call_node = node.child(0)?;
    let _ = try_parse_grammar_name(&call_node, "call")?;
    let func_name_node = call_node.child(0)?;
    let function_name = parser.get_text(&func_name_node);
    let parameters_arguments = func_name_node.next_sibling()?;
    let _ = try_parse_grammar_name(&parameters_arguments, "arguments")?;
    let parameters = parse_parameters(parser, &parameters_arguments);
    Some((function_name, parameters))
}

fn parse_parameters(parser: &Parser, node: &Node) -> Vec<Expression> {
    let mut result = Vec::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let grammar_name = child.grammar_name();

        if [",", "(", ")"].contains(&grammar_name) {
            continue;
        }

        result.push(parse_expression(parser, &child))
    }

    result
}

fn try_parse_do_block(parser: &Parser, node: &Node) -> Option<Expression> {
    let _ = try_parse_grammar_name(node, "do_block")?;
    let child = node.child(0)?;
    let _ = try_parse_grammar_name(&child, "do")?;
    let mut body = parse_sibilings_until_end(parser, &child)?;
    let body = normalize_block(&mut body);
    Some(body)
}

fn parse_sibilings_until_end(parser: &Parser, node: &Node) -> Option<Vec<Expression>> {
    let mut result = Vec::new();
    let mut next = node.next_sibling();

    while next.is_some() {
        let node = next?;
        if node.grammar_name() == "end" {
            return Some(result);
        }

        result.push(parse_expression(parser, &node));
        next = node.next_sibling();
    }

    Some(result)
}

fn try_parse_grammar_name<'a>(node: &'a Node, target: &str) -> Option<&'a Node<'a>> {
    if node.grammar_name() == target {
        Some(node)
    } else {
        None
    }
}

fn try_parse_either_identifier<'a>(
    parser: &Parser,
    node: &'a Node,
    target: &[&'a str],
) -> Option<&'a str> {
    for t in target {
        if try_parse_specific_identifier(parser, node, t).is_some() {
            return Some(t);
        }
    }

    None
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

fn try_parse_alias(parser: &Parser, node: &Node) -> Option<String> {
    if node.grammar_name() == "alias" {
        Some(parser.get_text(node))
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

fn print_expression(ast: &Expression, depth: usize) {
    match ast {
        Expression::Unparsed(expr, children) => {
            println!("{}{}└ {}", "  ".repeat(depth), depth, expr);

            for child in children {
                print_expression(child, depth + 1)
            }
        }
        Expression::Module(module) => {
            println!("{}{}└ {}", "  ".repeat(depth), depth, ast);
            print_expression(&module.body, depth + 1)
        }
        Expression::FunctionDef(function) => {
            println!("{}{}└ {}", "  ".repeat(depth), depth, ast);
            print_expression(&function.body, depth + 1)
        }
        Expression::Scope(scope) => {
            println!("{}{}└ {}", "  ".repeat(depth), depth, ast);

            for child in scope.body.clone() {
                print_expression(&child, depth + 1)
            }
        }
        expr => println!("{}{}└ {}", "  ".repeat(depth), depth, expr),
    }
}

#[cfg(test)]
mod tests {
    use super::Expression;
    use super::FunctionDef;
    use super::Module;
    use super::Parser;
    use super::Scope;

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
        ($x:expr, $children:expr) => {
            Expression::Unparsed($x.to_string(), $children)
        };
    }

    #[macro_export]
    macro_rules! scope {
        ($body:expr) => {
            Expression::Scope(Scope { body: $body })
        };
    }

    #[macro_export]
    macro_rules! parent {
        ($x:expr, $children:expr) => {
            Expression::Parent(Box::new($x), $children)
        };
    }

    #[macro_export]
    macro_rules! module {
        ($name:expr, $body:expr) => {
            Expression::Module(Module {
                name: $name.to_string(),
                body: Box::new($body),
            })
        };
    }

    // TODO: reuse code along these two def/defp macros
    #[macro_export]
    macro_rules! def {
        ($name:expr, $params:expr, $body:expr) => {
            Expression::FunctionDef(FunctionDef {
                is_private: false,
                name: $name.to_string(),
                parameters: $params,
                body: Box::new($body),
            })
        };
    }

    #[macro_export]
    macro_rules! defp {
        ($name:expr, $params:expr, $body:expr) => {
            Expression::FunctionDef(FunctionDef {
                is_private: true,
                name: $name.to_string(),
                parameters: $params,
                body: Box::new($body),
            })
        };
    }

    #[macro_export]
    macro_rules! string {
        ($s:expr) => {
            Expression::String($s.to_owned())
        };
    }

    #[macro_export]
    macro_rules! attribute {
        ($name:expr, $body:expr) => {
            Expression::Attribute($name.to_string(), Box::new($body))
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
            Expression::TreeSitterError(_, _) => {}
            _ => panic!("failed to match parent"),
        }
    }

    #[test]
    fn parse_empty_file() {
        let code = "";
        let result = parse(code);
        let expected = scope!(vec![]);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_module() {
        let code = "
        defmodule MyModule do
        end
        ";
        let result = parse(code);
        let expected = module!("MyModule", scope!(vec![]));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_single_line_blocks() {
        let code = "(a = 1; a + 1)";
        let result = parse(code);
        let expected = scope!(vec![
            unparsed!(
                "binary_operator",
                vec![
                    id!("a"),
                    unparsed!("=", vec![]),
                    unparsed!("integer", vec![])
                ]
            ),
            unparsed!(
                "binary_operator",
                vec![
                    id!("a"),
                    unparsed!("+", vec![]),
                    unparsed!("integer", vec![])
                ]
            ),
        ]);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_multi_line_blocks() {
        let code = "
        (
            a = 1
            a + 1
        )
        ";
        let result = parse(code);
        let expected = scope!(vec![
            unparsed!(
                "binary_operator",
                vec![
                    id!("a"),
                    unparsed!("=", vec![]),
                    unparsed!("integer", vec![])
                ]
            ),
            unparsed!(
                "binary_operator",
                vec![
                    id!("a"),
                    unparsed!("+", vec![]),
                    unparsed!("integer", vec![])
                ]
            ),
        ]);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_function_def() {
        let code = "
        def public(a, b) do
            a + b
        end
        ";
        let result = parse(code);
        let expected = def!(
            "public",
            vec![id!("a"), id!("b")],
            unparsed!(
                "binary_operator",
                vec![id!("a"), unparsed!("+", vec![]), id!("b")]
            )
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_function_defp() {
        let code = "
        defp my_func a, b do
            a + b
        end
        ";
        let result = parse(code);
        let expected = defp!(
            "my_func",
            vec![id!("a"), id!("b")],
            unparsed!(
                "binary_operator",
                vec![id!("a"), unparsed!("+", vec![]), id!("b")]
            )
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_attribute() {
        let code = r#"
        @doc "my docs"
        "#;
        let result = parse(code);
        let expected = attribute!("doc", string!("my docs"));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_simple_string() {
        let code = r#"
        "simple string!"
        "#;
        let result = parse(code);
        let expected = string!("simple string!");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_string_interpolation() {
        let code = r#"
        "today is #{weather}"
        "#;
        let result = parse(code);
        let expected = string!("today is #{weather}");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_string_escape_chars() {
        let code = "
        \"newline incoming\n\"
        ";
        let result = parse(code);
        let expected = string!("newline incoming\n");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_heredoc_string() {
        let code = r#"
        """
        Heredoc string!
        """
        "#;
        let result = parse(code);
        let expected = string!("\n        Heredoc string!\n        ");
        assert_eq!(result, expected)
    }

    // TODO: parse basic types like floats, ints, atoms
}

use string_interner::{DefaultSymbol, StringInterner};
use tree_sitter::{Node, Tree};

use crate::{
    indexer::{Index, Indexer},
    interner::intern_string,
};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Location {
    pub file: DefaultSymbol,
    pub line: usize,
    pub column: usize,
}

impl Default for Location {
    fn default() -> Self {
        let mut interner = StringInterner::default();

        Location {
            file: interner.get_or_intern("test"),
            line: 1,
            column: 0,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub name: DefaultSymbol,
    pub body: Box<Expression>,
    pub location: Location,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub is_private: bool,
    pub parameters: Vec<Expression>,
    pub body: Box<Expression>,
    pub location: Location,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Float(f64);

impl Eq for Float {}

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BinaryOperator {
    /// +
    Plus,
    /// -
    Minus,
    /// /
    Div,
    /// *
    Mult,
    /// |
    Union,
    /// **
    Pow,
    /// ++
    ListConcatenation,
    /// --
    ListSubtraction,
    /// and
    StrictAnd,
    /// &&
    RelaxedAnd,
    /// or
    StrictOr,
    /// ||
    RelaxedOr,
    /// in
    In,
    /// not in
    NotIn,
    /// <>
    BinaryConcat,
    /// |>
    Pipe,
    /// =~
    TextSearch,
    /// ==
    Equal,
    /// ===
    StrictEqual,
    /// !=
    NotEqual,
    /// !==
    StrictNotEqual,
    /// <
    LessThan,
    /// >
    GreaterThan,
    /// <=
    LessThanOrEqual,
    /// >=
    GreaterThanOrEqual,
    // =
    Match,
    // \\
    Default,
    // ::
    Type,
    ///  Operators that are parsed and can be overriden by libraries.
    ///  &&&
    ///  <<<
    ///  >>>
    ///  <<~
    ///  ~>>
    ///  <~
    ///  ~>
    ///  <~>
    ///  +++
    ///  ---
    Custom(String),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BinaryOperation {
    pub operator: BinaryOperator,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct List {
    pub body: Vec<Expression>,
    pub cons: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Module(Module),
    FunctionDef(FunctionDef),
    BinaryOperation(BinaryOperation),
    Attribute(String, Box<Expression>),
    Tuple(Vec<Expression>),
    List(List),
    Use(Vec<String>),
    Require(Vec<String>),
    Import(Vec<String>),
    Alias(Vec<String>),
    String(String),
    Integer(usize),
    Float(Float),
    Atom(String),
    Bool(bool),
    Block(Vec<Expression>),
    Identifier(String),
    Unparsed(String, Vec<Expression>),
    TreeSitterError(Location, Location),
}

#[derive(Debug)]
pub struct Parser {
    code: String,
    filepath: DefaultSymbol,
}

impl Parser {
    pub fn new(code: String, filepath: String) -> Self {
        let filepath = intern_string(&filepath);

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

        ast
    }

    pub fn index(&self, ast: &Expression) -> Vec<Index> {
        let mut modules = vec![];

        match ast {
            Expression::Block(ref expressions) => {
                for expression in expressions {
                    match expression {
                        Expression::Module(ref module) => {
                            let indexer = Indexer::new();
                            if let Some(module) = indexer.index(&module) {
                                modules.push(module)
                            }
                        }
                        _ => {}
                    }
                }
            }
            Expression::Module(ref module) => {
                let indexer = Indexer::new();
                if let Some(module) = indexer.index(&module) {
                    modules.push(module)
                }
            }
            _ => {}
        }

        modules
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
        Expression::Block(block.to_vec())
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
        "ERROR" => build_treesitter_error_node(parser, node),
        "identifier" => Expression::Identifier(parse_identifier_node(parser, node)),
        "call" => parse_call_node(parser, node),
        "string" => parse_string_node(parser, node),
        "atom" => parse_atom_node(parser, node),
        "tuple" => parse_tuple_node(parser, node),
        "list" => parse_list_node(parser, node),
        "binary_operator" => do_parse_binary_operation_node(parser, node),
        "alias" => parse_module_name_node(parser, node),
        "integer" => parse_integer_node(parser, node),
        "float" => parse_float_node(parser, node),
        "boolean" => parse_boolean_node(parser, node),
        "unary_operator" => parse_unary_operator(parser, node),
        "block" => parse_block_node(parser, node),
        _unknown => build_unparsed_node(parser, node),
    }
}

fn parse_tuple_node(parser: &Parser, node: &Node) -> Expression {
    let mut cursor = node.walk();
    let children = node.children(&mut cursor);

    let body = children
        .into_iter()
        .filter(|n| try_parse_either_grammar_name(&n, &["{", ",", "}"]).is_none())
        .map(|n| parse_expression(parser, &n))
        .collect::<Vec<_>>();

    Expression::Tuple(body)
}

fn parse_list_node(parser: &Parser, node: &Node) -> Expression {
    let mut cursor = node.walk();
    let mut list = List {
        body: vec![],
        cons: None,
    };

    for n in node.children(&mut cursor) {
        if let Some(_) = try_parse_either_grammar_name(&n, &["[", ",", "]"]) {
            continue;
        }

        match parse_expression(parser, &n) {
            Expression::BinaryOperation(BinaryOperation {
                left,
                operator: BinaryOperator::Union,
                right,
            }) => {
                list.body.push(*left);
                list.cons = Some(right);
            }
            expr => list.body.push(expr),
        }
    }

    Expression::List(list)
}

fn do_parse_binary_operation_node(parser: &Parser, node: &Node) -> Expression {
    parse_binary_operation_node(parser, node).unwrap_or_else(|| build_unparsed_node(parser, node))
}

fn parse_binary_operation_node(parser: &Parser, node: &Node) -> Option<Expression> {
    let left_node = node.child(0)?;
    let left = parse_expression(parser, &left_node);
    let operator_node = left_node.next_sibling()?;
    let operator_name = operator_node.grammar_name();

    let operator = match operator_name {
        "=" => BinaryOperator::Match,
        "+" => BinaryOperator::Plus,
        "-" => BinaryOperator::Minus,
        "*" => BinaryOperator::Mult,
        "**" => BinaryOperator::Pow,
        "::" => BinaryOperator::Type,
        "/" => BinaryOperator::Div,
        "++" => BinaryOperator::ListConcatenation,
        "--" => BinaryOperator::ListSubtraction,
        "and" => BinaryOperator::StrictAnd,
        "&&" => BinaryOperator::RelaxedAnd,
        "or" => BinaryOperator::StrictOr,
        "||" => BinaryOperator::RelaxedOr,
        "in" => BinaryOperator::In,
        "not in" => BinaryOperator::NotIn,
        "<>" => BinaryOperator::BinaryConcat,
        "|>" => BinaryOperator::Pipe,
        "|" => BinaryOperator::Union,
        "=~" => BinaryOperator::TextSearch,
        "==" => BinaryOperator::Equal,
        "===" => BinaryOperator::StrictEqual,
        "!=" => BinaryOperator::NotEqual,
        "!==" => BinaryOperator::StrictNotEqual,
        "<" => BinaryOperator::LessThan,
        ">" => BinaryOperator::GreaterThan,
        "<=" => BinaryOperator::LessThanOrEqual,
        ">=" => BinaryOperator::GreaterThanOrEqual,
        r#"\\"# => BinaryOperator::Default,
        other => BinaryOperator::Custom(other.to_owned()),
    };

    let right_node = operator_node.next_sibling()?;
    let right = parse_expression(parser, &right_node);

    Some(Expression::BinaryOperation(BinaryOperation {
        operator,
        left: Box::new(left),
        right: Box::new(right),
    }))
}

fn parse_string_node(parser: &Parser, node: &Node) -> Expression {
    let mut cursor = node.walk();
    let children = node.children(&mut cursor);

    let string = children
        .into_iter()
        .filter(|n| !["\"", "\"\"\""].contains(&n.grammar_name()))
        .map(|n| parser.get_text(&n))
        .collect::<Vec<_>>()
        .join("");

    Expression::String(string)
}

fn parse_atom_node(parser: &Parser, node: &Node) -> Expression {
    let content = &parser.get_text(node)[1..];
    Expression::Atom(content.to_string())
}

fn parse_integer_node(parser: &Parser, node: &Node) -> Expression {
    let integer_text = &parser.get_text(node);
    let integer_text = integer_text.replace('_', "");

    let result = if let Some(stripped) = integer_text.strip_prefix("0b") {
        usize::from_str_radix(stripped, 2).unwrap_or(0)
    } else if let Some(stripped) = integer_text.strip_prefix("0x") {
        usize::from_str_radix(stripped, 16).unwrap_or(0)
    } else if let Some(stripped) = integer_text.strip_prefix("0o") {
        usize::from_str_radix(stripped, 8).unwrap_or(0)
    } else {
        integer_text.parse::<usize>().unwrap_or(0)
    };

    Expression::Integer(result)
}

fn parse_float_node(parser: &Parser, node: &Node) -> Expression {
    let float_text = &parser.get_text(node);
    let float_text = float_text.replace('_', "");
    let result = float_text.parse::<f64>().unwrap_or(0.0);
    Expression::Float(Float(result))
}

fn parse_module_name_node(parser: &Parser, node: &Node) -> Expression {
    let content = parser.get_text(node);
    Expression::Atom(content)
}

fn parse_unary_operator(parser: &Parser, node: &Node) -> Expression {
    try_parse_attribute(parser, node).unwrap_or_else(|| build_unparsed_node(parser, node))
}

fn parse_boolean_node(parser: &Parser, node: &Node) -> Expression {
    Expression::Bool(parser.get_text(node) == "true")
}

fn parse_identifier_node(parser: &Parser, node: &Node) -> String {
    parser.get_text(node).to_owned()
}

fn parse_call_node(parser: &Parser, node: &Node) -> Expression {
    try_parse_module(parser, node)
        .or_else(|| try_parse_function_definition(parser, node))
        .or_else(|| {
            try_parse_module_operator(parser, node, "alias")
                .or_else(|| try_parse_multi_module_operator(parser, node, "alias"))
                .map(|modules| Expression::Alias(modules))
        })
        .or_else(|| {
            try_parse_module_operator(parser, node, "import")
                .or_else(|| try_parse_multi_module_operator(parser, node, "import"))
                .map(|modules| Expression::Import(modules))
        })
        .or_else(|| {
            try_parse_module_operator(parser, node, "require")
                .or_else(|| try_parse_multi_module_operator(parser, node, "require"))
                .map(|modules| Expression::Require(modules))
        })
        .or_else(|| {
            try_parse_module_operator(parser, node, "use")
                .or_else(|| try_parse_multi_module_operator(parser, node, "use"))
                .map(|modules| Expression::Use(modules))
        })
        .unwrap_or_else(|| build_unparsed_node(parser, node))
}

fn try_parse_module_operator(parser: &Parser, node: &Node, keyword: &str) -> Option<Vec<String>> {
    let alias_node = expect_child(node, "identifier")?;
    try_parse_either_identifier(parser, &alias_node, &[keyword])?;

    let alias_args_node = expect_sibling(&alias_node, "arguments")?;
    let atom_node = expect_child(&alias_args_node, "alias")?;
    let content = parser.get_text(&atom_node);
    Some(vec![content])
}

fn try_parse_multi_module_operator(
    parser: &Parser,
    node: &Node,
    keyword: &str,
) -> Option<Vec<String>> {
    let alias_node = expect_child(node, "identifier")?;
    try_parse_either_identifier(parser, &alias_node, &[keyword])?;

    let alias_args_node = expect_sibling(&alias_node, "arguments")?;
    let dot_node = expect_child(&alias_args_node, "dot")?;
    let atom_node = expect_child(&dot_node, "alias")?;
    let inner_dot_node = expect_sibling(&atom_node, ".")?;
    let tuple_node = expect_sibling(&inner_dot_node, "tuple")?;
    let tuple = parse_tuple_node(parser, &tuple_node);
    let base_module = parser.get_text(&atom_node);

    let aliases = match tuple {
        Expression::Tuple(items) => items
            .into_iter()
            .filter_map(|i| match i {
                Expression::Atom(atom) => Some(format!("{}.{}", base_module, atom)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => todo!(),
    };

    Some(aliases)
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

    Expression::Block(result)
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
    let defmodule_node = node.child(0)?;
    let _ = try_parse_specific_identifier(parser, &defmodule_node, "defmodule")?;
    let name_args_node = defmodule_node.next_sibling()?;
    let _ = try_parse_grammar_name(&name_args_node, "arguments")?;
    let name_node = name_args_node.child(0)?;
    let module_name = try_parse_alias_grammar_name(parser, &name_node)?;
    let child = name_args_node.next_sibling()?;
    let body = try_parse_do_block(parser, &child)?;

    let module_name_interned = intern_string(&module_name);
    let module = Module {
        name: module_name_interned,
        body: Box::new(body),
        location: tree_sitter_location_to_point(parser, name_node.start_position()),
    };

    Some(Expression::Module(module))
}

fn try_parse_function_definition(parser: &Parser, node: &Node) -> Option<Expression> {
    let def_node = expect_child(node, "identifier")?;
    let caller = try_parse_either_identifier(parser, &def_node, &["def", "defp"])?;
    let is_private = caller == "defp";
    let def_arguments_node = expect_sibling(&def_node, "arguments")?;

    let arguments_node_child = def_arguments_node.child(0)?;
    let (function_name, parameters) = match arguments_node_child.grammar_name() {
        "identifier" => {
            let function_name = parser.get_text(&arguments_node_child);
            (function_name, vec![])
        }
        _ => {
            let func_name_node = expect_child(&arguments_node_child, "identifier")?;
            let function_name = parser.get_text(&func_name_node);
            let parameters_arguments = expect_sibling(&func_name_node, "arguments")?;
            let parameters = parse_parameters(parser, &parameters_arguments);
            (function_name, parameters)
        }
    };

    let body = match def_arguments_node.next_sibling() {
        Some(body_node) => try_parse_do_block(parser, &body_node)?,
        None => {
            let comma_node = expect_sibling(&arguments_node_child, ",")?;
            try_parse_do_keyword(parser, &comma_node)?
        }
    };

    let function = FunctionDef {
        location: tree_sitter_location_to_point(parser, def_arguments_node.start_position()),
        is_private,
        parameters,
        name: function_name.to_string(),
        body: Box::new(body),
    };
    Some(Expression::FunctionDef(function))
}

fn try_parse_do_keyword(parser: &Parser, body_node: &Node) -> Option<Expression> {
    try_parse_grammar_name(body_node, ",")?;
    let keywords_node = expect_sibling(body_node, "keywords")?;
    let pair_node = expect_child(&keywords_node, "pair")?;
    let keyword_node = expect_child(&pair_node, "keyword")?;
    let body_node = keyword_node.next_sibling()?;
    // TODO: maybe we should go back to inlining blocks for single-expressions blocks now that the
    // indexer structure changed
    Some(Expression::Block(vec![parse_expression(
        parser, &body_node,
    )]))
}

fn expect_child<'a>(node: &'a Node, target: &str) -> Option<Node<'a>> {
    let child = node.child(0)?;
    let _ = try_parse_grammar_name(&child, target)?;
    Some(child)
}

fn expect_sibling<'a>(node: &'a Node, target: &str) -> Option<Node<'a>> {
    let child = node.next_sibling()?;
    let _ = try_parse_grammar_name(&child, target)?;
    Some(child)
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
    try_parse_grammar_name(node, "do_block")?;
    let child = expect_child(node, "do")?;
    let body = parse_sibilings_until_end(parser, &child)?;
    Some(Expression::Block(body))
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

fn try_parse_either_grammar_name<'a>(node: &'a Node, target: &[&'a str]) -> Option<&'a str> {
    for t in target {
        if try_parse_grammar_name(node, t).is_some() {
            return Some(t);
        }
    }

    None
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

fn try_parse_alias_grammar_name(parser: &Parser, node: &Node) -> Option<String> {
    if node.grammar_name() == "alias" {
        Some(parser.get_text(node))
    } else {
        None
    }
}

fn build_treesitter_error_node(parser: &Parser, node: &Node) -> Expression {
    Expression::TreeSitterError(
        tree_sitter_location_to_point(parser, node.start_position()),
        tree_sitter_location_to_point(parser, node.end_position()),
    )
}

fn tree_sitter_location_to_point(parser: &Parser, ts_point: tree_sitter::Point) -> Location {
    Location {
        file: parser.filepath,
        line: ts_point.row,
        column: ts_point.column,
    }
}

#[cfg(test)]
mod tests {
    use crate::simple_parser::BinaryOperator;
    use crate::interner::intern_string;

    use pretty_assertions::assert_eq;

    use super::BinaryOperation;
    use super::Expression;
    use super::Float;
    use super::FunctionDef;
    use super::List;
    use super::Location;
    use super::Module;
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
        ($x:expr, $children:expr) => {
            Expression::Unparsed($x.to_string(), $children)
        };
    }

    #[macro_export]
    macro_rules! block {
        ( $( $arg:expr ),* ) => {
            Expression::Block(vec![$($arg.to_owned(),)*])
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
        ($name:expr, $body:expr, $loc:expr) => {
            Expression::Module(Module {
                location: $loc,
                name: crate::interner::intern_string($name),
                body: Box::new($body),
            })
        };
    }

    #[macro_export]
    macro_rules! loc {
        ($line:expr, $col:expr) => {{
            let filepath = { crate::interner::intern_string("nofile") };

            Location {
                file: filepath,
                line: $line,
                column: $col,
            }
        }};
    }

    // TODO: reuse code along these two def/defp macros
    #[macro_export]
    macro_rules! def {
        ($name:expr, $params:expr, $body:expr, $location:expr) => {
            Expression::FunctionDef(FunctionDef {
                location: $location,
                is_private: false,
                name: $name.to_string(),
                parameters: $params,
                body: Box::new($body),
            })
        };
    }

    #[macro_export]
    macro_rules! defp {
        ($name:expr, $params:expr, $body:expr, $location:expr) => {
            Expression::FunctionDef(FunctionDef {
                location: $location,
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
    macro_rules! bool {
        ($v:expr) => {
            Expression::Bool($v)
        };
    }

    #[macro_export]
    macro_rules! atom {
        ($v:expr) => {
            Expression::Atom($v.to_string())
        };
    }

    #[macro_export]
    macro_rules! int {
        ($v:expr) => {
            Expression::Integer($v)
        };
    }

    #[macro_export]
    macro_rules! float {
        ($v:expr) => {
            Expression::Float(Float($v))
        };
    }

    #[macro_export]
    macro_rules! attribute {
        ($name:expr, $body:expr) => {
            Expression::Attribute($name.to_string(), Box::new($body))
        };
    }

    #[macro_export]
    macro_rules! binary_operation {
        ($left:expr, $operator:expr, $right:expr) => {
            Expression::BinaryOperation(BinaryOperation {
                operator: $operator,
                left: Box::new($left),
                right: Box::new($right),
            })
        };
    }

    #[macro_export]
    macro_rules! alias {
         ( $( $arg:expr ),* ) => {
             Expression::Alias(vec![$($arg.to_owned(),)*])
         };
    }

    #[macro_export]
    macro_rules! import {
         ( $( $arg:expr ),* ) => {
             Expression::Import(vec![$($arg.to_owned(),)*])
         };
    }

    #[macro_export]
    macro_rules! require {
         ( $( $arg:expr ),* ) => {
             Expression::Require(vec![$($arg.to_owned(),)*])
         };
    }

    #[macro_export]
    macro_rules! use_ {
         ( $( $arg:expr ),* ) => {
             Expression::Use(vec![$($arg.to_owned(),)*])
         };
    }

    #[macro_export]
    macro_rules! tuple {
         ( $( $arg:expr ),* ) => {
             Expression::Tuple(vec![$($arg,)*])
         };
    }

    #[macro_export]
    macro_rules! list {
         ( $( $arg:expr ),* ) => {
             Expression::List(List {
                body: vec![$($arg,)*],
                cons: None
             })
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
        let expected = block!();
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_module() {
        let code = "
        defmodule MyModule do
        end
        ";
        let result = parse(code);
        let expected = module!("MyModule", block!(), loc!(1, 18));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_single_line_blocks() {
        let code = "(a = 1; a + 1)";
        let result = parse(code);
        let expected = block!(
            binary_operation!(id!("a"), BinaryOperator::Match, int!(1)),
            binary_operation!(id!("a"), BinaryOperator::Plus, int!(1))
        );
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
        let expected = block!(
            binary_operation!(id!("a"), BinaryOperator::Match, int!(1)),
            binary_operation!(id!("a"), BinaryOperator::Plus, int!(1))
        );
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
            block!(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            loc!(1, 12)
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
            block!(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            loc!(1, 13)
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_function_no_args() {
        let code = "
        defp my_func() do
            a + b
        end
        ";
        let result = parse(code);
        let expected = defp!(
            "my_func",
            vec![],
            block!(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            loc!(1, 13)
        );
        assert_eq!(result, expected);

        let code = "
        defp my_func do
            a + b
        end
        ";
        let result = parse(code);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_function_keyword_form() {
        let code = "
        defp my_func(a, b), do: a + b
        ";
        let result = parse(code);
        let expected = defp!(
            "my_func",
            vec![id!("a"), id!("b")],
            block!(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            loc!(1, 13)
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_function_keyword_form_no_args() {
        let code = "
        defp my_func(), do: a + b
        ";
        let result = parse(code);
        let expected = defp!(
            "my_func",
            vec![],
            block!(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            loc!(1, 13)
        );
        assert_eq!(result, expected);

        let code = "
        defp my_func(), do: a + b
        ";
        let result = parse(code);
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

    #[test]
    fn parse_boolean_true() {
        let code = "true";
        let result = parse(code);
        let expected = bool!(true);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_boolean_false() {
        let code = "false";
        let result = parse(code);
        let expected = bool!(false);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_atom() {
        let code = ":custom_atom";
        let result = parse(code);
        let expected = atom!("custom_atom");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_module_name() {
        let code = "ModuleName";
        let result = parse(code);
        let expected = atom!("ModuleName");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_integers() {
        let code = "10";
        let result = parse(code);
        let expected = int!(10);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_integers_underscored() {
        let code = "1_000";
        let result = parse(code);
        let expected = int!(1000);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_integers_hex() {
        let code = "0x777";
        let result = parse(code);
        let expected = int!(0x777);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_integers_binary() {
        let code = "0b011001";
        let result = parse(code);
        let expected = int!(0b011001);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_integers_octal() {
        let code = "0o721";
        let result = parse(code);
        let expected = int!(0o721);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_float() {
        let code = "10.0";
        let result = parse(code);
        let expected = float!(10.0);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_float_scientific() {
        let code = "0.1e4";
        let result = parse(code);
        let expected = float!(1000.0);
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_binary_match() {
        let code = "a = 10";
        let result = parse(code);
        let expected = binary_operation!(id!("a"), BinaryOperator::Match, int!(10));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_empty_tuple() {
        let code = r#"{}"#;
        let result = parse(code);
        let expected = tuple!();
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_tuple() {
        let code = r#"{:ok, :success}"#;
        let result = parse(code);
        let expected = tuple!(atom!("ok"), atom!("success"));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_tuple_trailing_comma() {
        let code = "{1, 2, 3,}";
        let result = parse(code);
        let expected = tuple!(int!(1), int!(2), int!(3));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_multi_alias() {
        let code = "alias MySystem.{Queries, Entities}";
        let result = parse(code);
        let expected = alias!("MySystem.Queries", "MySystem.Entities");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_alias() {
        let code = "alias MyModule";
        let result = parse(code);
        let expected = alias!("MyModule");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_import() {
        let code = "import Ecto.Query";
        let result = parse(code);
        let expected = import!("Ecto.Query");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_require() {
        let code = "require Logger";
        let result = parse(code);
        let expected = require!("Logger");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_use() {
        let code = "use Oban.Worker";
        let result = parse(code);
        let expected = use_!("Oban.Worker");
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_empty_list() {
        let code = "[]";
        let result = parse(code);
        let expected = list!();
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_list() {
        let code = "[:atom, 1, 2]";
        let result = parse(code);
        let expected = list!(atom!("atom"), int!(1), int!(2));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_list_trailing_comma() {
        let code = "[1, 2, 3, 4,]";
        let result = parse(code);
        let expected = list!(int!(1), int!(2), int!(3), int!(4));
        assert_eq!(result, expected)
    }

    #[test]
    fn parse_list_cons() {
        let code = "[:atom, 1, 2 | rest]";
        let result = parse(code);
        let expected = Expression::List(List {
            body: vec![atom!("atom"), int!(1), int!(2)],
            cons: Some(Box::new(id!("rest"))),
        });
        assert_eq!(result, expected)
    }

    // TODO: parse do block in keyword form for functions and other structures

    // TODO: parse maps
    // TODO: parse major control structures: if, cond, case, with

    // TODO: parse @type as TypeAttribute with expr body
    // TODO: parse calls
}

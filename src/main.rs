use core::fmt;
use std::io;

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
    parameters: Vec<Identifier>,
}

#[derive(Debug, PartialEq, Eq)]
struct Module {
    name: Atom,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
struct List {
    items: Vec<Expression>,
    cons: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq)]
struct Tuple {
    items: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
struct Map {
    entries: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq)]
struct Struct {
    name: Atom,
    entries: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq)]
struct Keyword {
    pairs: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq)]
struct Attribute {
    name: Identifier,
    value: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
struct Call {
    target: Identifier,
    remote_callee: Option<Identifier>,
    arguments: Vec<Expression>,
}

type ParserResult<T> = Result<(T, u64), ParseError>;
type Parser<'a, T> = fn(&'a str, &'a Vec<Node<'a>>, u64) -> ParserResult<T>;

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
///
/// Since range can actually have 3 operands if taking the step into account, we don't treat
/// it as a binary operator.
#[derive(Debug, PartialEq, Eq)]
enum BinaryOperator {
    /// +
    Plus,
    /// -
    Minus,
    /// /
    Div,
    /// *
    Mult,
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

#[derive(Debug, PartialEq, Eq)]
struct Range {
    start: Box<Expression>,
    end: Box<Expression>,
    step: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq)]
struct BinaryOperation {
    operator: BinaryOperator,
    left: Box<Expression>,
    right: Box<Expression>,
}

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
#[derive(Debug, PartialEq, Eq)]
enum UnaryOperator {
    /// not
    StrictNot,
    /// !
    RelaxedNot,
    /// +
    Plus,
    /// -
    Minus,
    /// ^
    Pin,
}

#[derive(Debug, PartialEq, Eq)]
struct UnaryOperation {
    operator: UnaryOperator,
    operand: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq)]
enum Expression {
    Bool(bool),
    Nil,
    String(String),
    Float(Float),
    Integer(i128),
    Atom(Atom),
    Map(Map),
    Struct(Struct),
    KeywordList(Keyword),
    List(List),
    Tuple(Tuple),
    Identifier(Identifier),
    Block(Vec<Expression>),
    Module(Module),
    Attribute(Attribute),
    FunctionDef(Function),
    BinaryOperation(BinaryOperation),
    UnaryOperation(UnaryOperation),
    Range(Range),
    // TODO: anonymous functions
    // TODO: function capture
    //
    // TODO: double check whether quote accepts compiler metadata like `quote [keep: true] do`
    // TODO: Quote(Block)
    // TODO: MacroDef(Macro),
    Call(Call),
}

// TODO: Create Parser struct abstraction to hold things such as code, filename
#[derive(Debug, Clone)]
struct ParseError {
    error_type: ErrorType,
    file: String,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone)]
enum ErrorType {
    UnexpectedToken(String, String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            ErrorType::UnexpectedToken(expected, actual) => write!(
                f,
                "{}:{}:{} expected to find '{}' token, but instead got '{}'.",
                self.file, self.line, self.column, expected, actual
            ),
        }
    }
}

fn get_tree(code: &str) -> Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_elixir::language())
        .expect("Error loading elixir grammar");
    parser.parse(code, None).unwrap()
}

fn main() {
    println!("Please type something, or x to escape:");
    let mut code = String::new();

    while code != "x" {
        code.clear();
        io::stdin().read_line(&mut code).unwrap();
        let result = parse(&code);
        dbg!(result);
    }
}

fn parse(code: &str) -> Result<Expression, ParseError> {
    let tree = get_tree(code);
    let root_node = tree.root_node();
    let mut nodes = vec![];
    flatten_node_children(code, root_node, &mut nodes);
    dbg!(&nodes);
    let tokens = nodes.clone();

    if tokens.len() == 0 {
        return Ok(Expression::Block(vec![]));
    }

    let (result, _) = parse_many(&code, &tokens, 0, try_parse_expression)?;
    Ok(Expression::Block(result))
}

fn flatten_node_children<'a>(code: &str, node: Node<'a>, vec: &mut Vec<Node<'a>>) {
    if node.child_count() > 0 {
        let mut walker = node.walk();

        for node in node.children(&mut walker) {
            if node.grammar_name() != "comment" {
                vec.push(node);
                flatten_node_children(code, node, vec);
            }
        }
    }
}

fn try_parse_module(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(Expression, u64), ParseError> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(&code, tokens, offset, "defmodule")?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (module_name, offset) = try_parse_alias(code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)?;

    let module = Expression::Module(Module {
        name: module_name,
        body: Box::new(do_block),
    });
    Ok((module, offset))
}

fn try_parse_function_definition(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    let (offset, is_private) = match try_parse_keyword(code, tokens, offset, "defp") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(code, tokens, offset, "def")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let arg_consumer = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "arguments");
    let offset = try_consume(code, tokens, offset, arg_consumer);

    let call_consumer = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "call");
    let offset = try_consume(code, tokens, offset, call_consumer);

    let (function_name, offset) = try_parse_identifier(&code, tokens, offset)?;

    let (parameters, offset) = match try_parse_parameters(code, tokens, offset) {
        Ok(result) => result,
        Err(_) => (vec![], offset),
    };

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)?;

    let function = Expression::FunctionDef(Function {
        name: function_name,
        block: Box::new(do_block),
        parameters,
        is_private,
    });

    Ok((function, offset))
}

fn try_parse_parameters(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Vec<Identifier>> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let sep_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    let (parameters, offset) =
        match try_parse_sep_by(code, tokens, offset, try_parse_identifier, sep_parser) {
            Ok((parameters, offset)) => (parameters, offset),
            Err(_) => (vec![], offset),
        };

    let offset = if has_parenthesis {
        let (_node, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    Ok((parameters, offset))
}

fn try_parse_grammar_name<'a>(
    tokens: &Vec<Node<'a>>,
    offset: u64,
    expected: &str,
) -> Result<(Node<'a>, u64), ParseError> {
    let token = tokens[offset as usize];
    let actual = token.grammar_name();

    if actual == expected {
        Ok((token, offset + 1))
    } else {
        Err(build_unexpected_token_error(expected, &token))
    }
}

fn try_parse_identifier<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> Result<(Identifier, u64), ParseError> {
    let token = tokens[offset as usize];
    let actual = token.grammar_name();

    if actual == "identifier" {
        let identifier_name = extract_node_text(code, token);
        Ok((Identifier(identifier_name), offset + 1))
    } else {
        Err(build_unexpected_token_error("identifier", &token))
    }
}

fn try_parse_keyword<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
    keyword: &str,
) -> ParserResult<Identifier> {
    let token = tokens[offset as usize];
    let identifier_name = extract_node_text(code, token);
    let grammar_name = token.grammar_name();

    if grammar_name == "identifier" && identifier_name == keyword {
        Ok((Identifier(identifier_name), offset + 1))
    } else {
        Err(build_unexpected_token_error(keyword, &token))
    }
}

fn try_consume<'a, T>(
    code: &'a str,
    tokens: &'a Vec<Node<'a>>,
    offset: u64,
    parser: Parser<'a, T>,
) -> u64 {
    match parser(code, tokens, offset) {
        Ok((_node, new_offset)) => new_offset,
        Err(_) => offset,
    }
}

fn try_parse_call(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(Expression, u64), ParseError> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    // these are optional (present in remote calls, not present for local calls)
    let (offset, has_dot) = match try_parse_grammar_name(tokens, offset, "dot") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let dot_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ".");
    let offset = try_consume(code, tokens, offset, dot_parser);

    let mut remote_callee = None;

    let offset = match try_parse_grammar_name(tokens, offset, "alias") {
        Ok((node, new_offset)) => {
            remote_callee = Some(Identifier(extract_node_text(code, node)));
            new_offset
        }
        Err(_) => offset,
    };

    let offset = if remote_callee.is_none() && has_dot {
        let (identifier, new_offset) = try_parse_identifier(code, tokens, offset)?;
        remote_callee = Some(identifier);
        new_offset
    } else {
        offset
    };

    let offset = try_consume(code, tokens, offset, dot_parser);
    let (identifier, offset) = try_parse_identifier(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (arguments, offset) = match parse_expressions_sep_by_comma(code, tokens, offset) {
        Ok((expressions, new_offset)) => (expressions, new_offset),
        Err(err) => {
            dbg!(err);
            (vec![], offset)
        }
    };

    let offset = if has_parenthesis {
        let (_node, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    Ok((
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
) -> ParserResult<Vec<Expression>> {
    let sep_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    try_parse_sep_by(code, tokens, offset, try_parse_expression, sep_parser)
}

fn try_parse_keyword_expressions(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Keyword> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "keywords")?;
    let sep_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    try_parse_sep_by(code, tokens, offset, try_parse_keyword_pair, sep_parser)
        .and_then(|(pairs, offset)| Ok((Keyword { pairs }, offset)))
}

fn try_parse_sep_by<'a, T, S>(
    code: &'a str,
    tokens: &'a Vec<Node<'a>>,
    offset: u64,
    item_parser: Parser<'a, T>,
    sep_parser: Parser<'a, S>,
) -> ParserResult<Vec<T>> {
    let (expr, offset) = item_parser(code, tokens, offset)?;
    let mut offset_mut = offset;
    let mut expressions = vec![expr];

    // TODO: Having to add these checks feel weird
    if offset == tokens.len() as u64 {
        return Ok((expressions, offset));
    }

    let mut next_sep = sep_parser(code, tokens, offset);

    while next_sep.is_ok() {
        let (_next_comma, offset) = next_sep.unwrap();

        // Err here means trailing separator
        let (expr, offset) = match item_parser(code, tokens, offset) {
            Ok(result) => result,
            Err(_) => return Ok((expressions, offset)),
        };

        expressions.push(expr);

        if offset == tokens.len() as u64 {
            return Ok((expressions, offset));
        }

        offset_mut = offset;
        next_sep = sep_parser(code, tokens, offset);
    }

    Ok((expressions, offset_mut))
}

fn try_parse_keyword_pair(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "pair")?;

    let (atom, offset) = match try_parse_grammar_name(tokens, offset, "keyword") {
        Ok((atom_node, offset)) => {
            let atom_text = extract_node_text(&code, atom_node);
            // transform atom string from `atom: ` into `atom`
            let clean_atom = atom_text
                .replace(" ", "")
                .strip_suffix(":")
                .unwrap()
                .to_string();

            (Atom(clean_atom), offset)
        }
        Err(_) => try_parse_quoted_keyword(code, tokens, offset)?,
    };

    let (value, offset) = try_parse_expression(code, tokens, offset)?;
    let pair = (Expression::Atom(atom), value);
    Ok((pair, offset))
}

fn try_parse_quoted_keyword(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Atom> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "quoted_keyword")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "\"")?;
    let (quoted_content, offset) = try_parse_quoted_content(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "\"")?;
    Ok((Atom(quoted_content), offset))
}

fn try_parse_do_block<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do")?;
    let (body, offset) =
        parse_many(code, tokens, offset, try_parse_expression).unwrap_or_else(|_| (vec![], offset));
    let (_, offset) = try_parse_grammar_name(tokens, offset, "end")?;
    Ok((Expression::Block(body), offset))
}

fn try_parse_atom(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "atom")?;
    let atom_string = extract_node_text(code, atom_node)[1..].to_string();
    Ok((Expression::Atom(Atom(atom_string)), offset))
}

fn try_parse_quoted_atom(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "quoted_atom")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, ":")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "\"")?;
    let (atom_content, offset) = try_parse_quoted_content(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "\"")?;
    Ok((Expression::Atom(Atom(atom_content)), offset))
}

fn try_parse_bool(tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "boolean")?;

    let (boolean_result, offset) = match try_parse_grammar_name(tokens, offset, "true") {
        Ok((_, offset)) => (true, offset),
        Err(_) => (false, offset),
    };

    let (boolean_result, offset) = if !boolean_result {
        let (_, offset) = try_parse_grammar_name(tokens, offset, "false")?;
        (false, offset)
    } else {
        (boolean_result, offset)
    };

    Ok((Expression::Bool(boolean_result), offset))
}

fn try_parse_nil(tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    // INFO: for some reason treesitter-elixir outputs nil twice with the same span for each nil,
    // so we consume both
    let (_, offset) = try_parse_grammar_name(tokens, offset, "nil")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "nil")?;
    Ok((Expression::Nil, offset))
}

fn try_parse_map(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Map> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "{")?;

    // TODO: why is this the only case with a very weird and different grammar_name than what's
    // shown on debug? dbg!() for the Node says it's map_content but it's rather _items_with_trailing_separator
    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "_items_with_trailing_separator")
    });

    let key_value_parser =
        |code, tokens, offset| try_parse_specific_binary_operator(code, tokens, offset, "=>");
    let comma_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    let (expression_pairs, offset) =
        try_parse_sep_by(code, tokens, offset, key_value_parser, comma_parser)
            .unwrap_or_else(|_| (vec![], offset));

    // Keyword-notation-only map
    let (keyword_pairs, offset) = match try_parse_keyword_expressions(code, tokens, offset) {
        Ok((keyword, new_offset)) => (keyword.pairs, new_offset),
        Err(_) => (vec![], offset),
    };

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;
    let map = Map { entries: pairs };
    Ok((map, offset))
}

fn try_parse_struct(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Struct> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "struct")?;
    let (struct_name, offset) = try_parse_alias(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "{")?;

    // TODO: why is this the only case with a very weird and different grammar_name than what's
    // shown on debug? dbg!() for the Node says it's map_content but it's rather _items_with_trailing_separator
    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "_items_with_trailing_separator")
    });

    // Keyword-notation-only map
    let (keyword_pairs, offset) = match try_parse_keyword_expressions(code, tokens, offset) {
        Ok((keyword, new_offset)) => (keyword.pairs, new_offset),
        Err(_) => (vec![], offset),
    };

    // let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;
    let s = Struct {
        name: struct_name,
        entries: keyword_pairs,
    };
    Ok((s, offset))
}

fn try_parse_list(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "list")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "[")?;

    let (expressions, offset) =
        parse_expressions_sep_by_comma(code, tokens, offset).unwrap_or_else(|_| (vec![], offset));

    let (expr_before_cons, cons_expr, offset) = match try_parse_list_cons(code, tokens, offset) {
        Ok(((expr_before_cons, cons_expr), offset)) => {
            (Some(expr_before_cons), Some(Box::new(cons_expr)), offset)
        }
        Err(_) => (None, None, offset),
    };

    let (keyword, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .and_then(|(keyword, offset)| Ok((Some(Expression::KeywordList(keyword)), offset)))
        .unwrap_or_else(|_| (None, offset));

    dbg!(&keyword);

    let (_, offset) = try_parse_grammar_name(tokens, offset, "]")?;
    let expressions = expressions
        .into_iter()
        .chain(expr_before_cons)
        .chain(keyword)
        .collect();

    let list = Expression::List(List {
        items: expressions,
        cons: cons_expr,
    });
    Ok((list, offset))
}

fn try_parse_list_cons(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (expr_before_cons, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "|")?;
    let (cons_expr, offset) = try_parse_expression(code, tokens, offset)?;
    Ok(((expr_before_cons, cons_expr), offset))
}

fn try_parse_specific_binary_operator(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
    expected_operator: &str,
) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (left, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, expected_operator)?;
    let (right, offset) = try_parse_expression(code, tokens, offset)?;
    Ok(((left, right), offset))
}

fn try_parse_attribute(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Attribute> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "@")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (attribute_name, offset) = try_parse_identifier(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (value, offset) = try_parse_expression(code, tokens, offset)?;

    Ok((
        Attribute {
            name: attribute_name,
            value: Box::new(value),
        },
        offset,
    ))
}

fn try_parse_tuple(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(Tuple, u64), ParseError> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "tuple")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "{")?;
    let (expressions, offset) = match parse_expressions_sep_by_comma(code, tokens, offset) {
        Ok(result) => result,
        Err(_) => (vec![], offset),
    };
    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;

    let tuple = Tuple { items: expressions };
    Ok((tuple, offset))
}

fn try_parse_integer(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(i128, u64), ParseError> {
    let (integer_node, offset) = try_parse_grammar_name(tokens, offset, "integer")?;
    let integer_text = extract_node_text(code, integer_node);
    let integer_text = integer_text.replace("_", "");

    let int = integer_text.parse::<i128>().unwrap();
    Ok((int, offset))
}

fn try_parse_float(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(Float, u64), ParseError> {
    let (float_node, offset) = try_parse_grammar_name(tokens, offset, "float")?;
    let float_text = extract_node_text(code, float_node);

    let f = float_text.parse::<f64>().unwrap();
    Ok((Float(f), offset))
}

fn try_parse_string(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(String, u64), ParseError> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "string")?;
    let quotes = vec!["\"", "\"\"\""];
    let (_, offset) = try_parse_either_grammar_name(tokens, offset, &quotes)?;
    let (string_text, offset) = try_parse_quoted_content(code, tokens, offset)?;
    let (_, offset) = try_parse_either_grammar_name(tokens, offset, &quotes)?;
    Ok((string_text, offset))
}

/// Consumes everything from a quoted_content and returns it as a concatenated string.
/// It concatenates escape sequences and interpolations into the string
fn try_parse_quoted_content(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<String> {
    parse_many(code, tokens, offset, |code, tokens, offset| {
        let (string_node, offset) = try_parse_grammar_name(tokens, offset, "quoted_content")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "escape_sequence"))
            .or_else(|_| try_parse_interpolation(code, tokens, offset))?;

        let string_text = extract_node_text(code, string_node);
        Ok((string_text, offset))
    })
    .and_then(|(strings, offset)| Ok((strings.join(""), offset)))
}

// interpolations are discarded for now
// TODO: do not discard interpolations
fn try_parse_interpolation<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> ParserResult<Node<'a>> {
    let (interpolation_node, offset) = try_parse_grammar_name(tokens, offset, "interpolation")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "#{")?;
    let (_, offset) = try_parse_expression(code, tokens, offset)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;
    Ok((interpolation_node, offset))
}

fn try_parse_either_grammar_name<'a>(
    tokens: &Vec<Node<'a>>,
    offset: u64,
    grammar_names: &Vec<&str>,
) -> Result<(Node<'a>, u64), ParseError> {
    for grammar_name in grammar_names {
        match try_parse_grammar_name(tokens, offset, grammar_name) {
            Ok(result) => return Ok(result),
            Err(_) => {}
        }
    }

    let token = tokens[offset as usize];
    let expected = grammar_names.join(", ");
    Err(build_unexpected_token_error(&expected, &token))
}

/// Grammar_name = alias means module names in the TS elixir grammar.
/// e.g: ThisIsAnAlias, Elixir.ThisIsAlsoAnAlias
/// as these are also technically atoms, we return them as atoms
fn try_parse_alias(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Atom> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "alias")?;
    let atom_string = extract_node_text(code, atom_node);
    Ok((Atom(atom_string), offset))
}

fn try_parse_expression<'a>(
    code: &str,
    tokens: &Vec<Node<'a>>,
    offset: u64,
) -> ParserResult<Expression> {
    try_parse_module(code, tokens, offset)
        .or_else(|_| try_parse_function_definition(code, tokens, offset))
        .or_else(|_| try_parse_call(code, tokens, offset))
        .or_else(|_| try_parse_do_block(code, tokens, offset))
        .or_else(|_| {
            try_parse_alias(code, tokens, offset)
                .and_then(|(atom, offset)| Ok((Expression::Atom(atom), offset)))
        })
        .or_else(|_| {
            try_parse_string(code, tokens, offset)
                .and_then(|(string, offset)| Ok((Expression::String(string), offset)))
        })
        .or_else(|_| try_parse_nil(tokens, offset))
        .or_else(|_| try_parse_bool(tokens, offset))
        .or_else(|_| try_parse_atom(code, tokens, offset))
        .or_else(|_| try_parse_quoted_atom(code, tokens, offset))
        .or_else(|_| try_parse_list(code, tokens, offset))
        .or_else(|_| {
            try_parse_tuple(code, tokens, offset)
                .and_then(|(tuple, offset)| Ok((Expression::Tuple(tuple), offset)))
        })
        .or_else(|_| {
            try_parse_map(code, tokens, offset)
                .and_then(|(map, offset)| Ok((Expression::Map(map), offset)))
        })
        .or_else(|_| try_parse_unary_operator(code, tokens, offset))
        .or_else(|_| try_parse_binary_operator(code, tokens, offset))
        .or_else(|_| try_parse_range(code, tokens, offset))
        .or_else(|_| {
            try_parse_struct(code, tokens, offset)
                .and_then(|(s, offset)| Ok((Expression::Struct(s), offset)))
        })
        .or_else(|_| {
            try_parse_identifier(code, tokens, offset)
                .and_then(|(identifier, offset)| Ok((Expression::Identifier(identifier), offset)))
        })
        .or_else(|_| {
            try_parse_integer(code, tokens, offset)
                .and_then(|(integer, offset)| Ok((Expression::Integer(integer), offset)))
        })
        .or_else(|_| {
            try_parse_attribute(code, tokens, offset)
                .and_then(|(attribute, offset)| Ok((Expression::Attribute(attribute), offset)))
        })
        .or_else(|_| {
            try_parse_float(code, tokens, offset)
                .and_then(|(float, offset)| Ok((Expression::Float(float), offset)))
        })
}

fn try_parse_unary_operator(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (operator, offset) = try_parse_unary_operator_node(tokens, offset)?;
    let (operand, offset) = try_parse_expression(code, tokens, offset)?;
    let operation = Expression::UnaryOperation(UnaryOperation {
        operator,
        operand: Box::new(operand),
    });
    Ok((operation, offset))
}

fn try_parse_binary_operator(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (left, offset) = try_parse_expression(code, tokens, offset)?;
    let (operator, offset) = try_parse_binary_operator_node(tokens, offset)?;
    let (right, offset) = try_parse_expression(code, tokens, offset)?;
    let operation = Expression::BinaryOperation(BinaryOperation {
        operator,
        left: Box::new(left),
        right: Box::new(right),
    });
    Ok((operation, offset))
}

fn try_parse_unary_operator_node(tokens: &Vec<Node>, offset: u64) -> ParserResult<UnaryOperator> {
    let token = tokens[offset as usize];
    let operator = token.grammar_name();
    match operator {
        "+" => Ok((UnaryOperator::Plus, offset + 1)),
        "-" => Ok((UnaryOperator::Minus, offset + 1)),
        "!" => Ok((UnaryOperator::RelaxedNot, offset + 1)),
        "not" => Ok((UnaryOperator::StrictNot, offset + 1)),
        "^" => Ok((UnaryOperator::Pin, offset + 1)),
        _ => Err(build_unexpected_token_error("unary_operator", &token)),
    }
}

fn try_parse_binary_operator_node(tokens: &Vec<Node>, offset: u64) -> ParserResult<BinaryOperator> {
    let token = tokens[offset as usize];
    let operator = token.grammar_name();
    match operator {
        "=" => Ok((BinaryOperator::Match, offset + 1)),
        "+" => Ok((BinaryOperator::Plus, offset + 1)),
        "-" => Ok((BinaryOperator::Minus, offset + 1)),
        "*" => Ok((BinaryOperator::Mult, offset + 1)),
        "/" => Ok((BinaryOperator::Div, offset + 1)),
        "++" => Ok((BinaryOperator::ListConcatenation, offset + 1)),
        "--" => Ok((BinaryOperator::ListSubtraction, offset + 1)),
        "and" => Ok((BinaryOperator::StrictAnd, offset + 1)),
        "&&" => Ok((BinaryOperator::RelaxedAnd, offset + 1)),
        "or" => Ok((BinaryOperator::StrictOr, offset + 1)),
        "||" => Ok((BinaryOperator::RelaxedOr, offset + 1)),
        "in" => Ok((BinaryOperator::In, offset + 1)),
        "not in" => Ok((BinaryOperator::NotIn, offset + 1)),
        "<>" => Ok((BinaryOperator::BinaryConcat, offset + 1)),
        "|>" => Ok((BinaryOperator::Pipe, offset + 1)),
        "=~" => Ok((BinaryOperator::TextSearch, offset + 1)),
        "==" => Ok((BinaryOperator::Equal, offset + 1)),
        "===" => Ok((BinaryOperator::StrictEqual, offset + 1)),
        "!=" => Ok((BinaryOperator::NotEqual, offset + 1)),
        "!==" => Ok((BinaryOperator::StrictNotEqual, offset + 1)),
        "<" => Ok((BinaryOperator::LessThan, offset + 1)),
        ">" => Ok((BinaryOperator::GreaterThan, offset + 1)),
        "<=" => Ok((BinaryOperator::LessThanOrEqual, offset + 1)),
        ">=" => Ok((BinaryOperator::GreaterThanOrEqual, offset + 1)),
        unknown_operator => try_parse_custom_operator(&token, unknown_operator)
            .and_then(|op| Ok((BinaryOperator::Custom(op), offset + 1))),
    }
}

fn try_parse_range(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;

    let (offset, has_step) = match try_parse_grammar_name(tokens, offset, "binary_operator") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (start, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "..")?;
    let (end, offset) = try_parse_expression(code, tokens, offset)?;

    let (step, offset) = if has_step {
        let (_, offset) = try_parse_grammar_name(tokens, offset, "//")?;
        let (step, offset) = try_parse_expression(code, tokens, offset)?;
        (Some(Box::new(step)), offset)
    } else {
        (None, offset)
    };

    let range = Expression::Range(Range {
        start: Box::new(start),
        end: Box::new(end),
        step,
    });

    Ok((range, offset))
}

fn try_parse_custom_operator(token: &Node, operator_string: &str) -> Result<String, ParseError> {
    let custom_operators = vec![
        "&&&", "<<<", ">>>", "<<~", "~>>", "<~", "~>", "<~>", "+++", "---",
    ];

    if custom_operators.contains(&operator_string) {
        Ok(operator_string.to_string())
    } else {
        Err(build_unexpected_token_error("binary_operator", &token))
    }
}

fn build_unexpected_token_error(expected: &str, actual_token: &Node) -> ParseError {
    let point = actual_token.start_position();

    ParseError {
        file: "nofile".to_string(), // TODO: return file from here
        line: point.row,
        column: point.column,
        error_type: ErrorType::UnexpectedToken(
            expected.to_string(),
            actual_token.grammar_name().to_string(),
        ),
    }
}

fn parse_many<'a, T>(
    code: &'a str,
    tokens: &'a Vec<Node>,
    offset: u64,
    parser: Parser<'a, T>,
) -> ParserResult<Vec<T>> {
    let (expression, offset) = parser(code, tokens, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    if offset == tokens.len() as u64 {
        return Ok((expressions, offset_mut));
    }

    loop {
        match parser(code, tokens, offset_mut) {
            Ok((expression, offset)) => {
                expressions.push(expression);
                offset_mut = offset;

                if offset == tokens.len() as u64 {
                    return Ok((expressions, offset_mut));
                }
            }
            Err(_) => return Ok((expressions, offset_mut)),
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

        let target = Block(vec![Expression::Atom(Atom("atom".to_string()))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_atom_with_quotes() {
        let code = r#":"mywebsite@is_an_atom.com""#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Atom(Atom(
            "mywebsite@is_an_atom.com".to_string(),
        ))]);

        assert_eq!(result, target);
    }

    // TODO: parse charlists

    #[test]
    fn parse_quoted_atom_with_interpolation() {
        let code = ":\"john #{name}\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Atom(Atom("john #{name}".to_string()))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_quoted_atom_with_escape_sequence() {
        let code = ":\"john\n\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Atom(Atom("john\n".to_string()))]);

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

    // TODO: test function call with keyword arguments
    // TODO: parse anonymous function call: foo.()

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

    #[test]
    fn parse_hex_integer() {
        let code = "0x1F";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Integer(31)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_number() {
        let code = "0b1010";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Integer(10)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_octal_number() {
        let code = "0o777";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Integer(511)]);

        assert_eq!(result, target);
    }

    // TODO: parse integer in scientific notation
    // 1.0e-10

    #[test]
    fn parse_string() {
        let code = r#""string!""#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::String("string!".to_string())]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_string_escape_sequence() {
        let code = "\"hello world!\n different\nlines\nhere\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::String(
            "hello world!\n different\nlines\nhere".to_string(),
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_string_interpolation() {
        let code = r#""this is #{string_value}!""#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::String(
            "this is #{string_value}!".to_string(),
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_multiline_string() {
        let code = r#"
        """
        Multiline!
        String!
        """
        "#;
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
                    parameters: vec![],
                    block: Box::new(Block(vec![Expression::Call(Call {
                        target: Identifier("priv_func".to_string()),
                        remote_callee: None,
                        arguments: vec![],
                    })])),
                }),
                Expression::FunctionDef(Function {
                    name: Identifier("priv_func".to_string()),
                    is_private: true,
                    parameters: vec![],
                    block: Box::new(Block(vec![Expression::Integer(10)])),
                }),
            ])),
        })]);

        assert_eq!(result, target);
    }

    // TODO: parse functions with parameters
    // TODO: parse functions with default values parameters (\\)

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
            parameters: vec![],
            block: Box::new(Block(vec![Expression::Call(Call {
                target: Identifier("priv_func".to_string()),
                remote_callee: None,
                arguments: vec![],
            })])),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_no_parameters_parenthesis() {
        let code = "
        def func() do
        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            block: Box::new(Block(vec![])),
            parameters: vec![],
        })]);

        assert_eq!(result, target);
    }

    // TODO: parse default value parameters (\\)
    #[test]
    fn parse_function_with_parameters() {
        let code = "
        def func(a, b, c) do
        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            block: Box::new(Block(vec![])),
            parameters: vec![
                Identifier("a".to_string()),
                Identifier("b".to_string()),
                Identifier("c".to_string()),
            ],
        })]);

        assert_eq!(result, target);

        let code = "
        def func a, b, c do
        end
        ";
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    // TODO: parse function guard expression if any
    // TODO: parse function guard expression in oneliner form

    // #[test]
    // fn parse_function_oneliner() {
    //     // TODO: before adding this one, add suport to parsing keyword lists
    //     // Update try_parse_block to support do notation using keyword list
    //     let code = "
    //     # defmodule A, do: def a, do: 10
    //     def func, do: priv_func()
    //     ";
    //     let result = parse(&code).unwrap();
    //     let target = Block(vec![Expression::FunctionDef(Function {
    //         name: Identifier("func".to_string()),
    //         is_private: false,
    //         block: Box::new(Block(vec![Expression::Call(Call {
    //             target: Identifier("priv_func".to_string()),
    //             remote_callee: None,
    //             arguments: vec![],
    //         })])),
    //     })]);

    //     assert_eq!(result, target);
    // }

    // TODO: add @doc to function struct as a field to show in lsp hover

    #[test]
    fn parse_list() {
        let code = "
        [a, 10, :atom]
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::List(List {
            items: vec![
                Expression::Identifier(Identifier("a".to_string())),
                Expression::Integer(10),
                Expression::Atom(Atom("atom".to_string())),
            ],
            cons: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_list_with_keyword() {
        let code = "
        [a, b, module: :atom,   number: 200]
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::List(List {
            items: vec![
                Expression::Identifier(Identifier("a".to_string())),
                Expression::Identifier(Identifier("b".to_string())),
                Expression::KeywordList(Keyword {
                    pairs: vec![
                        (
                            Expression::Atom(Atom("module".to_string())),
                            Expression::Atom(Atom("atom".to_string())),
                        ),
                        (
                            Expression::Atom(Atom("number".to_string())),
                            Expression::Integer(200),
                        ),
                    ],
                }),
            ],
            cons: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_list() {
        let code = "[]";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::List(List {
            items: vec![],
            cons: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_list_cons() {
        let code = "[head, 10 | _tail] = [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Match,
            left: Box::new(Expression::List(List {
                items: vec![
                    Expression::Identifier(Identifier("head".to_string())),
                    Expression::Integer(10),
                ],
                cons: Some(Box::new(Expression::Identifier(Identifier(
                    "_tail".to_string(),
                )))),
            })),
            right: Box::new(Expression::List(List {
                items: vec![
                    Expression::Integer(1),
                    Expression::Integer(2),
                    Expression::Integer(3),
                ],
                cons: None,
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_tuple() {
        let code = "
        {a, 10, :atom}
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Tuple(Tuple {
            items: vec![
                Expression::Identifier(Identifier("a".to_string())),
                Expression::Integer(10),
                Expression::Atom(Atom("atom".to_string())),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_tuple() {
        let code = "{}";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Tuple(Tuple { items: vec![] })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_true() {
        let code = "true";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Bool(true)]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_false() {
        let code = "false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Bool(false)]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_nil() {
        let code = "nil";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Nil]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_map_atom_keys() {
        let code = "
        %{a: 10, b: true, c:    nil}
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map {
            entries: vec![
                (
                    Expression::Atom(Atom("a".to_string())),
                    Expression::Integer(10),
                ),
                (
                    Expression::Atom(Atom("b".to_string())),
                    Expression::Bool(true),
                ),
                (Expression::Atom(Atom("c".to_string())), Expression::Nil),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_map_expression_keys() {
        let code = r#"
        %{"a" => 10, "String" => true, "key" =>    nil, 10 => 30}
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map {
            entries: vec![
                (Expression::String("a".to_owned()), Expression::Integer(10)),
                (
                    Expression::String("String".to_owned()),
                    Expression::Bool(true),
                ),
                (Expression::String("key".to_owned()), Expression::Nil),
                (Expression::Integer(10), Expression::Integer(30)),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_map_mixed_keys() {
        let code = r#"
        %{"a" => 10, "String" => true, key: 50, Map: nil}
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map {
            entries: vec![
                (Expression::String("a".to_owned()), Expression::Integer(10)),
                (
                    Expression::String("String".to_owned()),
                    Expression::Bool(true),
                ),
                (
                    Expression::Atom(Atom("key".to_owned())),
                    Expression::Integer(50),
                ),
                (Expression::Atom(Atom("Map".to_owned())), Expression::Nil),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_map_quoted_atom_keys() {
        let code = r#"
        %{"a": 10, "myweb@site.com":   true}
        "#;

        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map {
            entries: vec![
                (
                    Expression::Atom(Atom("a".to_string())),
                    Expression::Integer(10),
                ),
                (
                    Expression::Atom(Atom("myweb@site.com".to_string())),
                    Expression::Bool(true),
                ),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_map() {
        let code = "%{}";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map { entries: vec![] })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_ignores_comments() {
        let code = "
        %{
            # this comment should not be a node in our tree-sitter
            # tree and parsing this should work fine!
        }
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Map(Map { entries: vec![] })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_attribute() {
        let code = "@timeout 5_000";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Attribute(Attribute {
            name: Identifier("timeout".to_string()),
            value: Box::new(Expression::Integer(5000)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_attributes_inside_module() {
        let code = r#"
        defmodule MyModule do
            @moduledoc """
            This is a nice module!
            """

            @doc """
            This is a nice function!
            """
            def func() do
                10
            end
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Module(Module {
            name: Atom("MyModule".to_string()),
            body: Box::new(Expression::Block(vec![
                Expression::Attribute(Attribute {
                    name: Identifier("moduledoc".to_string()),
                    value: Box::new(Expression::String(
                        "\n            This is a nice module!\n            ".to_string(),
                    )),
                }),
                Expression::Attribute(Attribute {
                    name: Identifier("doc".to_string()),
                    value: Box::new(Expression::String(
                        "\n            This is a nice function!\n            ".to_string(),
                    )),
                }),
                Expression::FunctionDef(Function {
                    name: Identifier("func".to_string()),
                    is_private: false,
                    block: Box::new(Expression::Block(vec![Expression::Integer(10)])),
                    parameters: vec![],
                }),
            ])),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_block_for_empty_input() {
        let code = "";
        let result = parse(&code).unwrap();
        let target = Block(vec![]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_struct() {
        let code = r#"
        %MyApp.User{name: "john", age: 25}
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Struct(Struct {
            name: Atom("MyApp.User".to_string()),
            entries: vec![
                (
                    Expression::Atom(Atom("name".to_string())),
                    Expression::String("john".to_string()),
                ),
                (
                    Expression::Atom(Atom("age".to_string())),
                    Expression::Integer(25),
                ),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_plus() {
        let code = "+10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::Plus,
            operand: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_minus() {
        let code = "-1000";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::Minus,
            operand: Box::new(Expression::Integer(1000)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_strict_not() {
        let code = "not true";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::StrictNot,
            operand: Box::new(Expression::Bool(true)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_relaxed_not() {
        let code = "!false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::RelaxedNot,
            operand: Box::new(Expression::Bool(false)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_chained_unary_operators() {
        let code = "!!false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::RelaxedNot,
            operand: Box::new(Expression::UnaryOperation(UnaryOperation {
                operator: UnaryOperator::RelaxedNot,
                operand: Box::new(Expression::Bool(false)),
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_pin() {
        let code = "^var";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::Pin,
            operand: Box::new(Expression::Identifier(Identifier("var".to_string()))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_plus() {
        let code = "20 + 40";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Plus,
            left: Box::new(Expression::Integer(20)),
            right: Box::new(Expression::Integer(40)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_minus() {
        let code = "100 - 50";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Minus,
            left: Box::new(Expression::Integer(100)),
            right: Box::new(Expression::Integer(50)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_star() {
        let code = "8 * 8";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Mult,
            left: Box::new(Expression::Integer(8)),
            right: Box::new(Expression::Integer(8)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_div() {
        let code = "10 / 2";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Div,
            left: Box::new(Expression::Integer(10)),
            right: Box::new(Expression::Integer(2)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_list_concatenation() {
        let code = "[1, 2, 3] ++ tail";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::ListConcatenation,
            left: Box::new(Expression::List(List {
                items: vec![
                    Expression::Integer(1),
                    Expression::Integer(2),
                    Expression::Integer(3),
                ],
                cons: None,
            })),
            right: Box::new(Expression::Identifier(Identifier("tail".to_string()))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_list_subtraction() {
        let code = "cache -- [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::ListSubtraction,
            left: Box::new(Expression::Identifier(Identifier("cache".to_string()))),
            right: Box::new(Expression::List(List {
                items: vec![
                    Expression::Integer(1),
                    Expression::Integer(2),
                    Expression::Integer(3),
                ],
                cons: None,
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_and() {
        let code = "true and false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::StrictAnd,
            left: Box::new(Expression::Bool(true)),
            right: Box::new(Expression::Bool(false)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_relaxed_and() {
        let code = ":atom && false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::RelaxedAnd,
            left: Box::new(Expression::Atom(Atom("atom".to_string()))),
            right: Box::new(Expression::Bool(false)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_or() {
        let code = "false or true";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::StrictOr,
            left: Box::new(Expression::Bool(false)),
            right: Box::new(Expression::Bool(true)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_relaxed_or() {
        let code = "a || b";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::RelaxedOr,
            left: Box::new(Expression::Identifier(Identifier("a".to_string()))),
            right: Box::new(Expression::Identifier(Identifier("b".to_string()))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_in() {
        let code = "1 in [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::In,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::List(List {
                items: vec![
                    Expression::Integer(1),
                    Expression::Integer(2),
                    Expression::Integer(3),
                ],
                cons: None,
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_not_in() {
        let code = "1 not in [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::NotIn,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::List(List {
                items: vec![
                    Expression::Integer(1),
                    Expression::Integer(2),
                    Expression::Integer(3),
                ],
                cons: None,
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_concat() {
        let code = r#""hello" <> "world""#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::BinaryConcat,
            left: Box::new(Expression::String("hello".to_string())),
            right: Box::new(Expression::String("world".to_string())),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_pipe() {
        let code = "value |> function()";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Pipe,
            left: Box::new(Expression::Identifier(Identifier("value".to_string()))),
            right: Box::new(Expression::Call(Call {
                target: Identifier("function".to_string()),
                remote_callee: None,
                arguments: vec![],
            })),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_equal() {
        let code = "10 == 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Equal,
            left: Box::new(Expression::Integer(10)),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_equal() {
        let code = "10.0 === 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::StrictEqual,
            left: Box::new(Expression::Float(Float(10.0))),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_less_than() {
        let code = "9 < 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::LessThan,
            left: Box::new(Expression::Integer(9)),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_less_than_or_equal() {
        let code = "9 <= 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::LessThanOrEqual,
            left: Box::new(Expression::Integer(9)),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_greater_than() {
        let code = "10 > 5";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::GreaterThan,
            left: Box::new(Expression::Integer(10)),
            right: Box::new(Expression::Integer(5)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_greater_than_or_equal() {
        let code = "2 >= 2";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::GreaterThanOrEqual,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(2)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_custom_binary_operator() {
        let custom_operators = vec![
            "&&&", "<<<", ">>>", "<<~", "~>>", "<~", "~>", "<~>", "+++", "---",
        ];

        for operator in custom_operators {
            let code = format!("2 {} 2", operator);
            let result = parse(&code).unwrap();

            let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
                operator: BinaryOperator::Custom(operator.to_string()),
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Integer(2)),
            })]);

            assert_eq!(result, target);
        }
    }

    #[test]
    fn parse_simple_range() {
        let code = "0..10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Range(Range {
            start: Box::new(Expression::Integer(0)),
            end: Box::new(Expression::Integer(10)),
            step: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_range_with_step() {
        let code = "0..10//2";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Range(Range {
            start: Box::new(Expression::Integer(0)),
            end: Box::new(Expression::Integer(10)),
            step: Some(Box::new(Expression::Integer(2))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_identifier() {
        let code = "my_var = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Match,
            left: Box::new(Expression::Identifier(Identifier("my_var".to_string()))),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_expressions() {
        let code = "10 = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Match,
            left: Box::new(Expression::Integer(10)),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_wildcard() {
        let code = "_ = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::BinaryOperation(BinaryOperation {
            operator: BinaryOperator::Match,
            left: Box::new(Expression::Identifier(Identifier("_".to_string()))),
            right: Box::new(Expression::Integer(10)),
        })]);

        assert_eq!(result, target);
    }
}

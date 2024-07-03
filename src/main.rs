use core::fmt;
use std::fmt::Debug;
use std::{env, ffi::OsStr, fs, path::Path};

use tree_sitter::{Node, Tree};
use walkdir::WalkDir;

#[derive(Debug, PartialEq, Eq, Clone)]
struct Identifier(String);

#[derive(Debug, PartialEq, Clone)]
struct Float(f64);

impl Eq for Float {}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Atom(String);

#[derive(Debug, PartialEq, Eq, Clone)]
struct Parameter {
    expression: Box<Expression>,
    default: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Function {
    name: Identifier,
    is_private: bool,
    body: Box<Expression>,
    guard_expression: Option<Box<Expression>>,
    parameters: Vec<Parameter>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Macro {
    name: Identifier,
    is_private: bool,
    body: Box<Expression>,
    guard_expression: Option<Box<Expression>>,
    parameters: Vec<Parameter>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Module {
    name: Atom,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct List {
    items: Vec<Expression>,
    cons: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Tuple {
    items: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Map {
    entries: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Struct {
    name: Atom,
    entries: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Attribute {
    name: Identifier,
    value: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Call {
    target: Box<Expression>,
    remote_callee: Option<Box<Expression>>,
    arguments: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Quote {
    options: Option<List>,
    body: Box<Expression>,
}

type ParserResult<T> = Result<(T, u64), ParseError>;
type Parser<'a, T> = fn(&'a str, &'a Vec<Node<'a>>, u64) -> ParserResult<T>;

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
///
/// Since range can actually have 3 operands if taking the step into account, we don't treat
/// it as a binary operator.
#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, PartialEq, Eq, Clone)]
struct Range {
    start: Box<Expression>,
    end: Box<Expression>,
    step: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct BinaryOperation {
    operator: BinaryOperator,
    left: Box<Expression>,
    right: Box<Expression>,
}

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, PartialEq, Eq, Clone)]
struct UnaryOperation {
    operator: UnaryOperator,
    operand: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct ConditionalExpression {
    condition_expression: Box<Expression>,
    do_expression: Box<Expression>,
    else_expression: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct CaseExpression {
    target_expression: Box<Expression>,
    arms: Vec<CaseArm>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct CondExpression {
    arms: Vec<CondArm>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct CondArm {
    condition: Box<Expression>,
    body: Box<Expression>,
}

// TODO: try to generalize this "stab" AST structure
#[derive(Debug, PartialEq, Eq, Clone)]
struct CaseArm {
    left: Box<Expression>,
    body: Box<Expression>,
    guard_expr: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Require {
    target: Atom,
    options: Option<List>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Access {
    target: Box<Expression>,
    access_expression: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct DotAccess {
    body: Box<Expression>,
    key: Identifier,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Alias {
    target: Atom,
    // TODO: maybe make empty list by default instead of None
    options: Option<List>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Use {
    target: Atom,
    // TODO: maybe make empty list by default instead of None
    options: Option<List>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Import {
    target: Atom,
    // TODO: maybe make empty list by default instead of None
    options: Option<List>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum FunctionCaptureRemoteCallee {
    Variable(Identifier),
    Module(Atom),
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct FunctionCapture {
    arity: usize,
    callee: Identifier,
    remote_callee: Option<FunctionCaptureRemoteCallee>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct LambdaClause {
    arguments: Vec<Expression>,
    guard: Option<Box<Expression>>,
    body: Expression,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Lambda {
    clauses: Vec<LambdaClause>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Sigil {
    name: String,
    content: String,
    modifier: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Expression {
    Bool(bool),
    Nil,
    String(String),
    Float(Float),
    Integer(usize),
    Atom(Atom),
    Map(Map),
    Struct(Struct),
    List(List),
    Tuple(Tuple),
    Identifier(Identifier),
    Block(Vec<Expression>),
    Module(Module),
    AttributeDef(Attribute),
    AttributeRef(Identifier),
    FunctionDef(Function),
    BinaryOperation(BinaryOperation),
    UnaryOperation(UnaryOperation),
    Range(Range),
    Require(Require),
    Alias(Alias),
    Import(Import),
    Use(Use),
    If(ConditionalExpression),
    Unless(ConditionalExpression),
    Case(CaseExpression),
    Cond(CondExpression),
    Access(Access),
    DotAccess(DotAccess),
    FunctionCapture(FunctionCapture),
    Capture(Box<Expression>),
    Lambda(Lambda),
    Grouping(Box<Expression>),
    Sigil(Sigil),
    // TODO: try catch
    // TODO: receive after
    Quote(Quote),
    MacroDef(Macro),
    Call(Call),
}

// TODO: Create Parser struct abstraction to hold things such as code, filename
#[derive(Debug, Clone)]
struct Point {
    line: usize,
    column: usize,
}

// TODO: Create Parser struct abstraction to hold things such as code, filename
#[derive(Debug, Clone)]
struct ParseError {
    error_type: ErrorType,
    start: Point,
    end: Point,
    offset: u64,
    file: String,
}

#[derive(Debug, Clone)]
enum ErrorType {
    UnexpectedToken(String),
    UnexpectedKeyword(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            ErrorType::UnexpectedToken(token) => write!(
                f,
                "{}:{}:{} unexpected token '{}'.",
                self.file, self.start.line, self.start.column, token
            ),
            ErrorType::UnexpectedKeyword(actual) => write!(
                f,
                "{}:{}:{} unexpected keyword: '{}'.",
                self.file, self.start.line, self.start.column, actual
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
    let args: Vec<String> = env::args().collect();

    use std::time::Instant;
    let now = Instant::now();

    {
        walkdir(&args[1]);
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // let file_path = args[1].clone();
    // let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    // let result = parse(&contents);
    // dbg!(&result);

    // println!("Please type something, or x to escape:");
    // let mut code = String::new();

    //     while code != "x" {
    //         code.clear();
    //         io::stdin().read_line(&mut code).unwrap();
    //         let result = parse(&code);
    //         dbg!(result);
    //     }
}

fn walkdir(path: &str) {
    let mut file_paths = Vec::new();

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(extension) = path.extension().and_then(OsStr::to_str) {
                match extension {
                    "ex" | "exs" => file_paths.push(path.to_owned()),
                    _ => (),
                }
            }
        }
    }

    use rayon::prelude::*;

    let results = file_paths
        .par_iter()
        .map(|path| (path, parse_file(path)))
        .collect::<Vec<_>>();

    let successes = results.iter().filter(|(_path, e)| e.is_ok());
    let failures = results.iter().filter(|(_path, e)| e.is_err());

    for (path, _file) in successes.clone() {
        println!("succesfully parsed {:?}", path);
    }

    for (path, _file) in failures.clone() {
        println!("failed to parsed {:?}", path);
    }

    println!(
        "finished indexing: errors: {} successes: {}",
        failures.count(),
        successes.count()
    )
}

fn parse_file(file_path: &Path) -> Result<Expression, ParseError> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    parse(&contents)
}

fn parse(code: &str) -> Result<Expression, ParseError> {
    let tree = get_tree(code);
    let root_node = tree.root_node();
    let mut nodes = vec![];
    flatten_node_children(root_node, &mut nodes);

    #[cfg(test)]
    dbg!(&nodes);

    let tokens = nodes.clone();

    let (result, _) = parse_all_tokens(code, &tokens, 0, try_parse_expression)?;

    Ok(Expression::Block(result))
}

fn flatten_node_children<'a>(node: Node<'a>, vec: &mut Vec<Node<'a>>) {
    if node.child_count() > 0 {
        let mut walker = node.walk();

        for node in node.children(&mut walker) {
            if node.grammar_name() != "comment" {
                vec.push(node);
                flatten_node_children(node, vec);
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
    let (_, offset) = try_parse_keyword(code, tokens, offset, "defmodule")?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (module_name, offset) = try_parse_module_name(code, tokens, offset)?;

    let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
    let list = extract_list(do_block_kw);
    let do_block = keyword_fetch(&list, atom!("do")).unwrap();

    let module = Expression::Module(Module {
        name: module_name,
        body: Box::new(do_block),
    });
    Ok((module, offset))
}

fn try_parse_capture_expression_variable(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Identifier> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "&")?;
    let (node, offset) = try_parse_grammar_name(tokens, offset, "integer")?;
    let content = extract_node_text(code, &node);
    Ok((Identifier(format!("&{}", content)), offset))
}

fn try_parse_capture_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "&")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (body, offset) = try_parse_expression(code, tokens, offset)?;

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    let capture = Expression::Capture(Box::new(body));
    Ok((capture, offset))
}

fn try_parse_remote_function_capture(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "&")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "dot")?;
    let (remote_callee, offset) = try_parse_module_name(code, tokens, offset)
        .map(|(name, offset)| (FunctionCaptureRemoteCallee::Module(name), offset))
        .or_else(|_| {
            let (name, offset) = try_parse_identifier(code, tokens, offset)?;
            Ok((FunctionCaptureRemoteCallee::Variable(name), offset))
        })?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, ".")?;
    let (local_callee, offset) = try_parse_identifier(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "/")?;
    let (arity, offset) = try_parse_integer(code, tokens, offset)?;

    let capture = Expression::FunctionCapture(FunctionCapture {
        arity,
        callee: local_callee,
        remote_callee: Some(remote_callee),
    });

    Ok((capture, offset))
}

fn try_parse_local_function_capture(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "&")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (local_callee, offset) = try_parse_identifier(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "/")?;
    let (arity, offset) = try_parse_integer(code, tokens, offset)?;

    let capture = Expression::FunctionCapture(FunctionCapture {
        arity,
        callee: local_callee,
        remote_callee: None,
    });

    Ok((capture, offset))
}

fn try_parse_quote(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    // TODO: try_pares_do_block with oneline support should do the trick here for both cases
    try_parse_quote_oneline(code, tokens, offset)
        .or_else(|_| try_parse_quote_block(code, tokens, offset))
}

fn try_parse_quote_block(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "quote")?;

    let arguments_p = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "arguments");
    let offset = try_consume(code, tokens, offset, arguments_p);

    let (options, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .or_else(|_| try_parse_keyword_list(code, tokens, offset))
        .map(|(opts, offset)| (Some(opts), offset))
        .unwrap_or((None, offset));

    let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
    let list = extract_list(do_block_kw);
    let do_block = keyword_fetch(&list, atom!("do")).unwrap();

    let capture = Expression::Quote(Quote {
        options,
        body: Box::new(do_block),
    });

    Ok((capture, offset))
}

fn try_parse_quote_oneline(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (quote_node, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "quote")?;

    let arguments_p = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "arguments");
    let offset = try_consume(code, tokens, offset, arguments_p);

    let (mut options, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .or_else(|_| try_parse_keyword_list(code, tokens, offset))
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))?;

    let (has_another_pair, offset) = try_parse_grammar_name(tokens, offset, ",")
        .map(|(_, offset)| (true, offset))
        .unwrap_or((false, offset));

    let (block, offset) = if has_another_pair {
        let (mut expr, offset) = try_parse_keyword_expressions(code, tokens, offset)
            .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))?;

        try_extract_do_keyword(offset, expr.pop(), &quote_node)?
    } else {
        try_extract_do_keyword(offset, options.pop(), &quote_node)?
    };

    // If there was only one option and it was a `do:` keyword, there are no actual options
    let options = if options.is_empty() {
        None
    } else {
        let list = List {
            items: options
                .into_iter()
                .map(|(left, right)| {
                    Expression::Tuple(Tuple {
                        items: vec![left, right],
                    })
                })
                .collect::<Vec<_>>(),
            cons: None,
        };

        Some(list)
    };

    let quote = Expression::Quote(Quote {
        options,
        body: Box::new(block),
    });

    Ok((quote, offset))
}

fn try_extract_do_keyword(
    offset: u64,
    pair: Option<(Expression, Expression)>,
    start_node: &Node,
) -> ParserResult<Expression> {
    match pair {
        Some((Expression::Atom(atom), block)) => {
            if atom == Atom("do".to_string()) {
                Ok((block, offset))
            } else {
                Err(build_unexpected_token_error(offset, start_node))
            }
        }
        Some(_) | None => Err(build_unexpected_token_error(offset, start_node)),
    }
}

fn try_parse_keyword_list(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<List> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "list")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "[")?;
    let (options, offset) = try_parse_keyword_expressions(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "]")?;
    Ok((options, offset))
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

    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "arguments")
    });

    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "call")
    });

    if let Ok(((guard_expr, parameters, function_name), offset)) =
        try_parse_function_guard(code, tokens, offset)
    {
        let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
        let list = extract_list(do_block_kw);
        let do_block = keyword_fetch(&list, atom!("do")).unwrap();

        let function = Expression::FunctionDef(Function {
            name: function_name,
            body: Box::new(do_block),
            guard_expression: Some(Box::new(guard_expr)),
            parameters,
            is_private,
        });

        return Ok((function, offset));
    };

    let (function_name, offset) = try_parse_identifier(code, tokens, offset)?;

    let (parameters, offset) =
        try_parse_parameters(code, tokens, offset).unwrap_or((vec![], offset));

    let (is_keyword_form, offset) = try_parse_grammar_name(tokens, offset, ",")
        .map(|(_, offset)| (true, offset))
        .unwrap_or((false, offset));

    let (do_block, offset) = if is_keyword_form {
        let (mut keywords, offset) = try_parse_keyword_expressions(code, tokens, offset)
            .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))?;
        let first_pair = keywords.remove(0);
        // TODO: refactor this branch
        (first_pair.1, offset)
    } else {
        let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
        let list = extract_list(do_block_kw);
        let do_block = keyword_fetch(&list, atom!("do")).unwrap();
        (do_block, offset)
    };

    let function = Expression::FunctionDef(Function {
        name: function_name,
        body: Box::new(do_block),
        guard_expression: None,
        parameters,
        is_private,
    });

    Ok((function, offset))
}

fn try_parse_do_keyword(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (mut list, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .or_else(|_| try_parse_keyword_list(code, tokens, offset))?;

    let do_atom = atom!("do");

    match list.items.pop().unwrap() {
        Expression::Tuple(tuple) => {
            if tuple.items[0] == do_atom && tuple.items.len() == 2 {
                Ok((Expression::List(list), offset))
            } else {
                panic!("invalid tuple keyword was parsed")
            }
        }
        _ => panic!("invalid tuple keyword was parsed"),
    }
}

// TODO: generalize all of this function and macro definition code
fn try_parse_macro_definition(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;

    let (offset, is_private) = match try_parse_keyword(code, tokens, offset, "defmacrop") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(code, tokens, offset, "defmacro")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "arguments")
    });

    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "call")
    });

    if let Ok(((guard_expr, parameters, macro_name), offset)) =
        try_parse_function_guard(code, tokens, offset)
    {
        let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
        let list = extract_list(do_block_kw);
        let do_block = keyword_fetch(&list, atom!("do")).unwrap();

        let macro_def = Expression::MacroDef(Macro {
            name: macro_name,
            body: Box::new(do_block),
            guard_expression: Some(Box::new(guard_expr)),
            parameters,
            is_private,
        });

        return Ok((macro_def, offset));
    };

    let (macro_name, offset) = try_parse_identifier(code, tokens, offset)?;

    let (parameters, offset) =
        try_parse_parameters(code, tokens, offset).unwrap_or((vec![], offset));

    let (is_keyword_form, offset) = try_parse_grammar_name(tokens, offset, ",")
        .map(|(_, offset)| (true, offset))
        .unwrap_or((false, offset));

    let (do_block, offset) = if is_keyword_form {
        let (mut keywords, offset) = try_parse_keyword_expressions(code, tokens, offset)
            .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))?;
        let first_pair = keywords.remove(0);
        (first_pair.1, offset)
    } else {
        let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
        let list = extract_list(do_block_kw);
        let do_block = keyword_fetch(&list, atom!("do")).unwrap();
        (do_block, offset)
    };

    let macro_def = Expression::MacroDef(Macro {
        name: macro_name,
        body: Box::new(do_block),
        guard_expression: None,
        parameters,
        is_private,
    });

    Ok((macro_def, offset))
}

fn try_parse_function_guard(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<(Expression, Vec<Parameter>, Identifier)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (function_name, offset) = try_parse_identifier(code, tokens, offset)?;

    let (parameters, offset) = match try_parse_parameters(code, tokens, offset) {
        Ok(result) => result,
        Err(_) => (vec![], offset),
    };

    let (_, offset) = try_parse_grammar_name(tokens, offset, "when")?;
    let (guard_expression, offset) = try_parse_expression(code, tokens, offset)?;

    let comma_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    let offset = try_consume(code, tokens, offset, comma_parser);

    Ok(((guard_expression, parameters, function_name), offset))
}

fn try_parse_parameters(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Vec<Parameter>> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let sep_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");
    let (parameters, offset) =
        match try_parse_sep_by(code, tokens, offset, try_parse_parameter, sep_parser) {
            Ok((parameters, offset)) => (parameters, offset),
            Err(_) => (vec![], offset),
        };

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    Ok((parameters, offset))
}

fn try_parse_parameter(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Parameter> {
    match try_parse_parameter_default_value(code, tokens, offset) {
        Ok(result) => Ok(result),
        Err(_) => try_parse_expression(code, tokens, offset).map(|(expr, offset)| {
            let expr = Parameter {
                expression: Box::new(expr),
                default: None,
            };

            (expr, offset)
        }),
    }
}

fn try_parse_parameter_default_value(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Parameter> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (expression, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, r#"\\"#)?;
    let (default, offset) = try_parse_expression(code, tokens, offset)?;

    let parameter = Parameter {
        expression: Box::new(expression),
        default: Some(Box::new(default)),
    };

    Ok((parameter, offset))
}

fn try_parse_grammar_name<'a>(
    tokens: &Vec<Node<'a>>,
    offset: u64,
    expected: &str,
) -> ParserResult<Node<'a>> {
    if tokens.len() == offset as usize {
        let node = tokens.last().unwrap();

        return Err(build_unexpected_token_error(offset, node));
    }

    let token = tokens[offset as usize];
    let actual = token.grammar_name();

    if actual == expected {
        Ok((token, offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
    }
}

fn try_parse_identifier(code: &str, tokens: &[Node], offset: u64) -> ParserResult<Identifier> {
    let token = tokens[offset as usize];
    let actual = token.grammar_name();
    let identifier_name = extract_node_text(code, &token);

    let reserved = [
        "defp",
        "def",
        "defmodule",
        "import",
        "use",
        "require",
        "alias",
    ];
    if reserved.contains(&identifier_name.as_str()) {
        return Err(build_unexpected_keyword_error(offset, &token));
    }

    if actual == "identifier" {
        Ok((Identifier(identifier_name), offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
    }
}

fn try_parse_keyword(
    code: &str,
    tokens: &[Node],
    offset: u64,
    keyword: &str,
) -> ParserResult<Identifier> {
    let token = tokens[offset as usize];
    let identifier_name = extract_node_text(code, &token);
    let grammar_name = token.grammar_name();

    if grammar_name == "identifier" && identifier_name == keyword {
        Ok((Identifier(identifier_name), offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
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

fn try_parse_remote_call(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "dot")?;
    let (remote_callee, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, ".")?;
    let (local_callee, offset) = try_parse_identifier(code, tokens, offset)?;
    let (mut arguments, offset) = try_parse_call_arguments(code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)
        .map(|(block, offset)| (Some(block), offset))
        .unwrap_or((None, offset));

    arguments.extend(do_block);

    let call = Expression::Call(Call {
        target: Box::new(Expression::Identifier(local_callee)),
        remote_callee: Some(Box::new(remote_callee)),
        arguments,
    });

    Ok((call, offset))
}

fn try_parse_local_call(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (local_callee, offset) = try_parse_identifier(code, tokens, offset)?;
    let (mut arguments, offset) = try_parse_call_arguments(code, tokens, offset)?;

    let (do_block, offset) = try_parse_do_block(code, tokens, offset)
        .map(|(block, offset)| (Some(block), offset))
        .unwrap_or((None, offset));

    arguments.extend(do_block);

    let call = Expression::Call(Call {
        target: Box::new(Expression::Identifier(local_callee)),
        remote_callee: None,
        arguments,
    });

    Ok((call, offset))
}

fn try_parse_call_arguments(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Vec<Expression>> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(tokens, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (mut base_arguments, offset) = match parse_expressions_sep_by_comma(code, tokens, offset) {
        Ok((expressions, new_offset)) => (expressions, new_offset),
        Err(_) => (vec![], offset),
    };

    let (keyword_arguments, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .map(|(keyword, offset)| (Some(Expression::List(keyword)), offset))
        .unwrap_or((None, offset));

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(tokens, offset, ")")?;
        new_offset
    } else {
        offset
    };

    base_arguments.extend(keyword_arguments);
    Ok((base_arguments, offset))
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
) -> ParserResult<List> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "keywords")?;
    let sep_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");

    try_parse_sep_by(code, tokens, offset, try_parse_keyword_pair, sep_parser).map(
        |(pairs, offset)| {
            let tuples = pairs
                .into_iter()
                .map(|(key, value)| {
                    Expression::Tuple(Tuple {
                        items: vec![key, value],
                    })
                })
                .collect::<Vec<_>>();

            let list = List {
                items: tuples,
                cons: None,
            };

            (list, offset)
        },
    )
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
            let atom_text = extract_node_text(code, &atom_node);
            // transform atom string from `atom: ` into `atom`
            let clean_atom = atom_text
                .split_whitespace()
                .collect::<String>()
                .strip_suffix(':')
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

fn try_parse_cond_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "cond")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do")?;

    let end_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "end");

    let (arms, offset) =
        parse_until(code, tokens, offset, try_parse_stab, end_parser).map(|(arms, offset)| {
            let arms = arms
                .into_iter()
                .map(|(left, right)| CondArm {
                    condition: Box::new(left),
                    body: Box::new(right),
                })
                .collect();

            (arms, offset)
        })?;
    let (_, offset) = end_parser(code, tokens, offset)?;

    let case = CondExpression { arms };

    Ok((Expression::Cond(case), offset))
}

fn try_parse_access_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "access_call")?;
    let (target, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "[")?;
    let (access_expression, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "]")?;

    let access = Access {
        target: Box::new(target),
        access_expression: Box::new(access_expression),
    };

    Ok((Expression::Access(access), offset))
}

fn try_parse_case_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "case")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;

    let (case_expression, offset) = try_parse_expression(code, tokens, offset)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do")?;

    let end_parser = |_code, tokens, offset| try_parse_grammar_name(tokens, offset, "end");
    let (arms, offset) = parse_until(code, tokens, offset, try_parse_case_arm, end_parser)?;
    let (_, offset) = end_parser(code, tokens, offset)?;

    let case = CaseExpression {
        target_expression: Box::new(case_expression),
        arms,
    };

    Ok((Expression::Case(case), offset))
}

fn try_parse_sigil(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "sigil")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "~")?;
    let (name_node, offset) = try_parse_grammar_name(tokens, offset, "sigil_name")?;
    let sigil_name = extract_node_text(code, &name_node);

    let sigil_open_delimiters = vec!["/", "|", "\"", "'", "(", " [", "{", "<"];
    let sigil_close_delimiters = vec!["/", "|", "\"", "'", ")", " ]", "}", ">"];

    let (_, offset) = try_parse_either_token(tokens, offset, &sigil_open_delimiters)?;

    let (sigil_content_node, offset) = try_parse_grammar_name(tokens, offset, "quoted_content")?;
    let sigil_content = extract_node_text(code, &sigil_content_node);

    let (_, offset) = try_parse_either_token(tokens, offset, &sigil_close_delimiters)?;

    let (modifier, offset) = try_parse_grammar_name(tokens, offset, "sigil_modifiers")
        .map(|(node, offset)| {
            let sigil_content = extract_node_text(code, &node);
            (Some(sigil_content), offset)
        })
        .unwrap_or((None, offset));

    let sigil = Sigil {
        name: sigil_name,
        content: sigil_content,
        modifier,
    };

    Ok((Expression::Sigil(sigil), offset))
}

// TODO: use in more places
fn try_parse_either_token<'a>(
    tokens: &'a Vec<Node>,
    offset: u64,
    allowed_tokens: &Vec<&'static str>,
) -> ParserResult<Node<'a>> {
    for token in allowed_tokens {
        if let Ok((node, offset)) = try_parse_grammar_name(tokens, offset, token) {
            return Ok((node, offset));
        }
    }

    Err(build_unexpected_token_error(
        offset,
        &tokens[offset as usize],
    ))
}

fn try_parse_lambda(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "anonymous_function")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "fn")?;

    let end_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, "end");

    let lambda_clause_parser = |code, tokens, offset| {
        try_parse_lambda_simple_clause(code, tokens, offset)
            .or_else(|_| try_parse_lambda_guard_clause(code, tokens, offset))
    };

    let (clauses, offset) = parse_until(code, tokens, offset, lambda_clause_parser, end_parser)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "end")?;

    let lambda = Lambda { clauses };

    Ok((Expression::Lambda(lambda), offset))
}

fn try_parse_lambda_simple_clause(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<LambdaClause> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "stab_clause")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let comma_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");

    let (arguments, offset) =
        try_parse_sep_by(code, tokens, offset, try_parse_expression, comma_parser)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "->")?;
    let (body, offset) = try_parse_stab_body(code, tokens, offset)?;

    let lambda_clause = LambdaClause {
        arguments,
        body,
        guard: None,
    };

    Ok((lambda_clause, offset))
}

fn try_parse_lambda_guard_clause(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<LambdaClause> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "stab_clause")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let comma_parser = |_, tokens, offset| try_parse_grammar_name(tokens, offset, ",");

    let (arguments, offset) =
        try_parse_sep_by(code, tokens, offset, try_parse_expression, comma_parser)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "when")?;
    let (guard, offset) = try_parse_expression(code, tokens, offset)?;

    let (_, offset) = try_parse_grammar_name(tokens, offset, "->")?;
    let (body, offset) = try_parse_stab_body(code, tokens, offset)?;

    let lambda_clause = LambdaClause {
        arguments,
        body,
        guard: Some(Box::new(guard)),
    };

    Ok((lambda_clause, offset))
}

fn try_parse_stab(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "stab_clause")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (left, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "->")?;
    let (body, offset) = try_parse_stab_body(code, tokens, offset)?;
    Ok(((left, body), offset))
}

fn try_parse_stab_body(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "body")?;

    let end_parser = |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "stab_clause")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "end"))
    };

    let (mut body, offset) = parse_until(code, tokens, offset, try_parse_expression, end_parser)?;
    let body = if body.len() == 1 {
        body.remove(0)
    } else {
        Expression::Block(body)
    };

    Ok((body, offset))
}

fn try_parse_case_arm(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<CaseArm> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "stab_clause")?;

    if let Ok(((left, right, guard), offset)) = try_parse_case_arm_guard(code, tokens, offset) {
        let case_arm = CaseArm {
            left: Box::new(left),
            body: Box::new(right),
            guard_expr: Some(Box::new(guard)),
        };

        return Ok((case_arm, offset));
    }

    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (left, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "->")?;
    let (body, offset) = try_parse_stab_body(code, tokens, offset)?;

    let case_arm = CaseArm {
        left: Box::new(left),
        body: Box::new(body),
        guard_expr: None,
    };

    Ok((case_arm, offset))
}

fn try_parse_case_arm_guard(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<(Expression, Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (left, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "when")?;
    let (guard_expression, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "->")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "body")?;
    let (body, offset) = try_parse_expression(code, tokens, offset)?;

    Ok(((left, body, guard_expression), offset))
}

// TODO: make another version of parse_do_block that has the same core logic but work specifically
// for keyword syntax

/// Parses a sugarized do block and returns a desugarized keyword structure
fn try_parse_do_block(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "do")?;

    // TODO: do a bit of cleanup on this parser
    let block_end_parser = |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "end")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "else_block"))
            .or_else(|_| try_parse_grammar_name(tokens, offset, "after_block"))
            .or_else(|_| try_parse_grammar_name(tokens, offset, "rescue_block"))
            .or_else(|_| try_parse_grammar_name(tokens, offset, "catch_block"))
    };

    let optional_block_end_parser =
        |_, tokens, offset| try_parse_grammar_name(tokens, offset, "end");

    let (mut body, offset) =
        parse_until(code, tokens, offset, try_parse_expression, block_end_parser)
            .unwrap_or((vec![], offset));

    let optional_block_parser = |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "else")
            .map(|(_, offset)| (atom!("else"), offset))
            .or_else(|_| {
                try_parse_grammar_name(tokens, offset, "after")
                    .map(|(_, offset)| (atom!("after"), offset))
            })
            .or_else(|_| {
                try_parse_grammar_name(tokens, offset, "rescue")
                    .map(|(_, offset)| (atom!("rescue"), offset))
            })
            .or_else(|_| {
                try_parse_grammar_name(tokens, offset, "catch")
                    .map(|(_, offset)| (atom!("catch"), offset))
            })
    };

    let optional_block_start_parser = |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "else_block")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "after_block"))
            .or_else(|_| try_parse_grammar_name(tokens, offset, "rescue_block"))
            .or_else(|_| try_parse_grammar_name(tokens, offset, "catch_block"))
    };

    let (optional_block, optional_block_atom, offset) =
        optional_block_start_parser(code, tokens, offset)
            .and_then(|(_, offset)| {
                let (atom, offset) = optional_block_parser(code, tokens, offset)?;

                let (else_block, offset) = parse_until(
                    code,
                    tokens,
                    offset,
                    try_parse_expression,
                    optional_block_end_parser,
                )?;

                Ok((Some(else_block), Some(atom), offset))
            })
            .unwrap_or((None, None, offset));

    let (_, offset) = try_parse_grammar_name(tokens, offset, "end")?;

    // TODO: move to a helper function
    let body = if body.len() == 1 {
        body.pop().unwrap()
    } else {
        Expression::Block(body)
    };

    let mut base_list = List {
        items: vec![tuple!(atom!("do"), body)],
        cons: None,
    };

    if let Some(mut block) = optional_block {
        let block = if block.len() == 1 {
            block.pop().unwrap()
        } else {
            Expression::Block(block)
        };

        let block_atom = optional_block_atom.unwrap();
        base_list.items.push(tuple!(block_atom, block));
    }

    Ok((Expression::List(base_list), offset))
}

fn try_parse_atom(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "atom")?;
    let atom_string = extract_node_text(code, &atom_node)[1..].to_string();
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
    let (keyword_pairs, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))
        .unwrap_or((vec![], offset));

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;
    let map = Map { entries: pairs };
    Ok((map, offset))
}

fn try_parse_struct(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Struct> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "struct")?;
    let (struct_name, offset) = try_parse_module_name(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "{")?;

    // TODO: why is this the only case with a very weird and different grammar_name than what's
    // shown on debug? dbg!() for the Node says it's map_content but it's rather _items_with_trailing_separator
    let offset = try_consume(code, tokens, offset, |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "_items_with_trailing_separator")
    });

    // Keyword-notation-only map
    let (keyword_pairs, offset) = try_parse_keyword_expressions(code, tokens, offset)
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))
        .unwrap_or((vec![], offset));

    // let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(tokens, offset, "}")?;
    let s = Struct {
        name: struct_name,
        entries: keyword_pairs,
    };
    Ok((s, offset))
}

fn convert_keyword_expression_lists_to_tuples(list: List) -> Vec<(Expression, Expression)> {
    list.items
        .into_iter()
        .map(|mut entry| extract_key_value_from_tuple(&mut entry))
        .collect::<Vec<_>>()
}

/// Assumes the received expression is a keyword tuple and extracts the data or crashes otherwise
fn extract_key_value_from_tuple(expr: &mut Expression) -> (Expression, Expression) {
    match expr {
        Expression::Tuple(tuple) => {
            let b = tuple.items.pop().unwrap();
            let a = tuple.items.pop().unwrap();
            (a, b)
        }
        _ => panic!("expecting tuple but got another expression"),
    }
}

fn try_parse_grouping(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "block")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "(")?;
    let (expression, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, ")")?;
    let grouping = Expression::Grouping(Box::new(expression));
    Ok((grouping, offset))
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
        .map(|(keyword, offset)| (Some(Expression::List(keyword)), offset))
        .unwrap_or_else(|_| (None, offset));

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

fn try_parse_attribute_definition(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Attribute> {
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

fn try_parse_attribute_reference(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Identifier> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "@")?;
    let (attribute_name, offset) = try_parse_identifier(code, tokens, offset)?;
    Ok((attribute_name, offset))
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
) -> Result<(usize, u64), ParseError> {
    let (integer_node, offset) = try_parse_grammar_name(tokens, offset, "integer")?;
    let integer_text = extract_node_text(code, &integer_node);
    let integer_text = integer_text.replace('_', "");

    let result = if let Some(stripped) = integer_text.strip_prefix("0b") {
        usize::from_str_radix(stripped, 2).unwrap()
    } else if let Some(stripped) = integer_text.strip_prefix("0x") {
        usize::from_str_radix(stripped, 16).unwrap()
    } else if let Some(stripped) = integer_text.strip_prefix("0o") {
        usize::from_str_radix(stripped, 8).unwrap()
    } else {
        integer_text.parse::<usize>().unwrap()
    };

    Ok((result, offset))
}

fn try_parse_float(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> Result<(Float, u64), ParseError> {
    let (float_node, offset) = try_parse_grammar_name(tokens, offset, "float")?;
    let float_text = extract_node_text(code, &float_node);
    let float_text = float_text.replace('_', "");

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
    let (_, offset) = try_parse_either_token(tokens, offset, &quotes)?;

    // empty string
    if let Ok((_, offset)) = try_parse_either_token(tokens, offset, &quotes) {
        return Ok((String::new(), offset));
    };

    let (string_text, offset) = try_parse_quoted_content(code, tokens, offset)?;
    let (_, offset) = try_parse_either_token(tokens, offset, &quotes)?;
    Ok((string_text, offset))
}

/// Consumes everything from a quoted_content and returns it as a concatenated string.
/// It concatenates escape sequences and interpolations into the string
fn try_parse_quoted_content(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<String> {
    let parser = |code, tokens, offset| {
        let (string_node, offset) = try_parse_grammar_name(tokens, offset, "quoted_content")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "escape_sequence"))
            .or_else(|_| try_parse_interpolation(code, tokens, offset))?;

        let string_text = extract_node_text(code, &string_node);
        Ok((string_text, offset))
    };

    let end_parser = |_, tokens, offset| {
        try_parse_grammar_name(tokens, offset, "\"")
            .or_else(|_| try_parse_grammar_name(tokens, offset, "\"\"\""))
    };

    parse_until(code, tokens, offset, parser, end_parser)
        .map(|(strings, offset)| (strings.join(""), offset))
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

/// Grammar_name = alias means module names in the TS elixir grammar.
/// e.g: ThisIsAnAlias, Elixir.ThisIsAlsoAnAlias
/// as these are also technically atoms, we return them as atoms
fn try_parse_module_name(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Atom> {
    let (atom_node, offset) = try_parse_grammar_name(tokens, offset, "alias")?;
    let atom_string = extract_node_text(code, &atom_node);
    Ok((Atom(atom_string), offset))
}

fn try_parse_expression(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    try_parse_module(code, tokens, offset)
        .or_else(|err| try_parse_grouping(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_remote_call(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_dot_access(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_function_definition(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_macro_definition(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_if_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_unless_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_case_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_cond_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_access_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_lambda(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_sigil(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_quote(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_remote_function_capture(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_local_function_capture(code, tokens, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_capture_expression_variable(code, tokens, offset)
                .map(|(var, offset)| (Expression::Identifier(var), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_capture_expression(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_alias(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_require(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_use(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_import(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_local_call(code, tokens, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_module_name(code, tokens, offset)
                .map(|(atom, offset)| (Expression::Atom(atom), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_string(code, tokens, offset)
                .map(|(string, offset)| (Expression::String(string), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_nil(tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_bool(tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_atom(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_quoted_atom(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_list(code, tokens, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_tuple(code, tokens, offset)
                .map(|(tuple, offset)| (Expression::Tuple(tuple), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_map(code, tokens, offset)
                .map(|(map, offset)| (Expression::Map(map), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_unary_operator(code, tokens, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_attribute_reference(code, tokens, offset)
                .map(|(attribute, offset)| (Expression::AttributeRef(attribute), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_binary_operator(code, tokens, offset).map_err(|_| err))
        .or_else(|err| try_parse_range(code, tokens, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_struct(code, tokens, offset)
                .map(|(s, offset)| (Expression::Struct(s), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_identifier(code, tokens, offset)
                .map(|(identifier, offset)| (Expression::Identifier(identifier), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_integer(code, tokens, offset)
                .map(|(integer, offset)| (Expression::Integer(integer), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_attribute_definition(code, tokens, offset)
                .map(|(attribute, offset)| (Expression::AttributeDef(attribute), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            // FIXME: currently we are losing the most specific parsing error somewhere in the stack
            // if you want to see the specific token that caused the error, uncomment the dbg below
            // and look at the logs
            // dbg!(&err);
            try_parse_float(code, tokens, offset)
                .map(|(float, offset)| (Expression::Float(float), offset))
                .map_err(|_| err)
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

fn try_parse_if_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "if")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (condition_expression, offset) = try_parse_expression(code, tokens, offset)?;

    let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
    let list = extract_list(do_block_kw);
    let do_block = keyword_fetch(&list, atom!("do")).unwrap();
    let else_block = keyword_fetch(&list, atom!("else")).map(Box::new);

    let if_expr = Expression::If(ConditionalExpression {
        condition_expression: Box::new(condition_expression),
        do_expression: Box::new(do_block),
        else_expression: else_block,
    });

    Ok((if_expr, offset))
}

fn try_parse_unless_expression(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, "unless")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (condition_expression, offset) = try_parse_expression(code, tokens, offset)?;

    let (do_block_kw, offset) = try_parse_do_block(code, tokens, offset)?;
    let list = extract_list(do_block_kw);
    let do_block = keyword_fetch(&list, atom!("do")).unwrap();
    let else_block = keyword_fetch(&list, atom!("else")).map(Box::new);

    let unless_expr = Expression::Unless(ConditionalExpression {
        condition_expression: Box::new(condition_expression),
        do_expression: Box::new(do_block),
        else_expression: else_block,
    });

    Ok((unless_expr, offset))
}

fn extract_list(list_expr: Expression) -> List {
    match list_expr {
        Expression::List(list) => list,
        _ => panic!("was expecting list expression, got another expression"),
    }
}

fn keyword_fetch(list: &List, target_key: Expression) -> Option<Expression> {
    for item in &list.items {
        if let Expression::Tuple(t) = item {
            if t.items[0] == target_key {
                return Some(t.items[1].clone());
            }
        }
    }

    None
}

fn try_parse_require(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let ((required_module, options), offset) =
        try_parse_module_operator(code, tokens, offset, "require")?;

    let require = Expression::Require(Require {
        target: required_module,
        options,
    });

    Ok((require, offset))
}

fn try_parse_alias(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let ((aliased_module, options), offset) =
        try_parse_module_operator(code, tokens, offset, "alias")?;

    let require = Expression::Alias(Alias {
        target: aliased_module,
        options,
    });

    Ok((require, offset))
}

fn try_parse_use(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let ((used_module, options), offset) = try_parse_module_operator(code, tokens, offset, "use")?;

    let require = Expression::Use(Use {
        target: used_module,
        options,
    });

    Ok((require, offset))
}

fn try_parse_dot_access(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "dot")?;
    let (body, offset) = try_parse_expression(code, tokens, offset)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, ".")?;
    let (identifier, offset) = try_parse_identifier(code, tokens, offset)?;

    let dot_access = Expression::DotAccess(DotAccess {
        body: Box::new(body),
        key: identifier,
    });

    Ok((dot_access, offset))
}

fn try_parse_import(code: &str, tokens: &Vec<Node>, offset: u64) -> ParserResult<Expression> {
    let ((imported_module, options), offset) =
        try_parse_module_operator(code, tokens, offset, "import")?;

    let require = Expression::Import(Import {
        target: imported_module,
        options,
    });

    Ok((require, offset))
}

/// Module operator == import, use, require, alias
fn try_parse_module_operator(
    code: &str,
    tokens: &Vec<Node>,
    offset: u64,
    target: &str,
) -> ParserResult<(Atom, Option<List>)> {
    let (_, offset) = try_parse_grammar_name(tokens, offset, "call")?;
    let (_, offset) = try_parse_keyword(code, tokens, offset, target)?;
    let (_, offset) = try_parse_grammar_name(tokens, offset, "arguments")?;
    let (target_module, offset) = try_parse_module_name(code, tokens, offset)?;

    let (options, offset) =
        // TODO: this only fails when the only content in the file is the require expression
        // find a general fix to these bounds check
        if offset == tokens.len() as u64 {
            (None, offset)
        } else {
            match try_parse_grammar_name(tokens, offset, ",") {
                Ok((_, offset)) => {
                    let (options, offset) = try_parse_keyword_expressions(code, tokens, offset)?;
                    (Some(options), offset)
                },
                Err(_) => (None, offset)
            }
        };

    Ok(((target_module, options), offset))
}

fn try_parse_unary_operator_node(tokens: &[Node], offset: u64) -> ParserResult<UnaryOperator> {
    let token = tokens[offset as usize];
    let operator = token.grammar_name();
    match operator {
        "+" => Ok((UnaryOperator::Plus, offset + 1)),
        "-" => Ok((UnaryOperator::Minus, offset + 1)),
        "!" => Ok((UnaryOperator::RelaxedNot, offset + 1)),
        "not" => Ok((UnaryOperator::StrictNot, offset + 1)),
        "^" => Ok((UnaryOperator::Pin, offset + 1)),
        _ => Err(build_unexpected_token_error(offset, &token)),
    }
}

fn try_parse_binary_operator_node(
    tokens: &[Node],
    offset: u64,
) -> ParserResult<BinaryOperator> {
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
        unknown_operator => try_parse_custom_operator(&token, offset, unknown_operator)
            .map(|op| (BinaryOperator::Custom(op), offset + 1)),
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

fn try_parse_custom_operator(
    token: &Node,
    offset: u64,
    operator_string: &str,
) -> Result<String, ParseError> {
    let custom_operators = [
        "&&&", "<<<", ">>>", "<<~", "~>>", "<~", "~>", "<~>", "+++", "---",
    ];

    if custom_operators.contains(&operator_string) {
        Ok(operator_string.to_string())
    } else {
        Err(build_unexpected_token_error(offset, token))
    }
}

fn build_unexpected_token_error(offset: u64, actual_token: &Node) -> ParseError {
    let start = actual_token.start_position();
    let end = actual_token.end_position();

    ParseError {
        offset,
        file: "nofile".to_string(), // TODO: return file from here
        start: Point {
            line: start.row,
            column: start.column,
        },
        end: Point {
            line: end.row,
            column: end.column,
        },
        error_type: ErrorType::UnexpectedToken(actual_token.grammar_name().to_string()),
    }
}

fn build_unexpected_keyword_error(offset: u64, actual_token: &Node) -> ParseError {
    let start = actual_token.start_position();
    let end = actual_token.end_position();

    ParseError {
        offset,
        file: "nofile".to_string(), // TODO: return file from here
        start: Point {
            line: start.row,
            column: start.column,
        },
        end: Point {
            line: end.row,
            column: end.column,
        },
        error_type: ErrorType::UnexpectedKeyword(actual_token.grammar_name().to_string()),
    }
}

fn parse_until<'a, T: Debug, E>(
    code: &'a str,
    tokens: &'a Vec<Node>,
    offset: u64,
    parser: Parser<'a, T>,
    end_parser: Parser<'a, E>,
) -> ParserResult<Vec<T>> {
    let (expression, offset) = parser(code, tokens, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    loop {
        match end_parser(code, tokens, offset_mut) {
            // Do not consume end token, return previous offset
            Ok((_, offset)) => return Ok((expressions, offset - 1)),
            Err(_) => {
                let (expression, new_offset) = parser(code, tokens, offset_mut)?;
                expressions.push(expression);
                offset_mut = new_offset;
            }
        }
    }
}

fn parse_all_tokens<'a, T: Debug>(
    code: &'a str,
    tokens: &'a Vec<Node>,
    offset: u64,
    parser: Parser<'a, T>,
) -> ParserResult<Vec<T>> {
    if tokens.is_empty() {
        return Ok((vec![], 0));
    }

    let (expression, offset) = parser(code, tokens, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    loop {
        if offset_mut == tokens.len() as u64 {
            return Ok((expressions, offset_mut));
        }

        let (expression, offset) = parser(code, tokens, offset_mut)?;
        expressions.push(expression);
        offset_mut = offset;
    }
}

fn extract_node_text(code: &str, node: &Node) -> String {
    code[node.byte_range()].to_string()
}

#[macro_export]
macro_rules! atom {
    ($x:expr) => {
        Expression::Atom(Atom($x.to_string()))
    };
}

#[macro_export]
macro_rules! nil {
    () => {
        Expression::Nil
    };
}

#[macro_export]
macro_rules! int {
    ($x:expr) => {
        Expression::Integer($x)
    };
}

#[macro_export]
macro_rules! float {
    ($x:expr) => {
        Expression::Float(Float($x))
    };
}

#[macro_export]
macro_rules! bool {
    ($x:expr) => {
        Expression::Bool($x)
    };
}

#[macro_export]
macro_rules! id {
    ($x:expr) => {
        Expression::Identifier(Identifier($x.to_string()))
    };
}

#[macro_export]
macro_rules! string {
    ($x:expr) => {
        Expression::String($x.to_string())
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
macro_rules! list {
    ( $( $item:expr ),* ) => {
        Expression::List(List {
            items: vec![$($item,)*],
            cons: None,
        })
    };
}

#[macro_export]
macro_rules! tuple {
    ( $( $item:expr ),* ) => {
        Expression::Tuple(Tuple {
            items: vec![$($item,)*],
        })
    };
}

#[macro_export]
macro_rules! list_cons {
    ( $items:expr, $cons:expr ) => {
        Expression::List(List {
            items: $items,
            cons: $cons,
        })
    };
}

#[macro_export]
macro_rules! call {
    ( $target:expr, $( $arg:expr ),* ) => {
        Expression::Call(Call {
            target: Box::new($target),
            remote_callee: None,
            arguments: vec![$($arg,)*]
        })
    };
}

#[cfg(test)]
mod tests {
    use super::Expression::Block;
    use super::*;

    #[test]
    fn parse_atom() {
        let code = ":atom";
        let result = parse(&code).unwrap();

        let target = Block(vec![atom!("atom")]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_atom_with_quotes() {
        let code = r#":"mywebsite@is_an_atom.com""#;
        let result = parse(&code).unwrap();

        let target = Block(vec![atom!("mywebsite@is_an_atom.com")]);

        assert_eq!(result, target);
    }

    // TODO: parse charlists

    #[test]
    fn parse_quoted_atom_with_interpolation() {
        let code = ":\"john #{name}\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![atom!("john #{name}")]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_quoted_atom_with_escape_sequence() {
        let code = ":\"john\n\"";
        let result = parse(&code).unwrap();

        let target = Block(vec![atom!("john\n")]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_alias() {
        let code = "Elixir.MyModule";
        let result = parse(&code).unwrap();

        let target = Block(vec![atom!("Elixir.MyModule")]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_identifier() {
        let code = "my_variable_name";
        let result = parse(&code).unwrap();

        let target = Block(vec![id!("my_variable_name")]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_identifier_attribute() {
        let code = "@my_module_attribute";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::AttributeRef(Identifier(
            "my_module_attribute".to_string(),
        ))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_remote_call_module() {
        let code = "Enum.reject(a, my_func)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("reject")),
            remote_callee: Some(Box::new(atom!("Enum"))),
            arguments: vec![id!("a"), id!("my_func")],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_remote_expression_call() {
        let code = r#":base64.encode("abc")"#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("encode")),
            remote_callee: Some(Box::new(atom!("base64"))),
            arguments: vec![Expression::String("abc".to_string())],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_remote_call_identifier() {
        let code = "json_parser.parse(body)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("parse".to_string())),
            remote_callee: Some(Box::new(id!("json_parser"))),
            arguments: vec![id!("body")],
        })]);

        assert_eq!(result, target);
    }

    // TODO: parse anonymous function call: foo.()

    #[test]
    fn parse_local_call() {
        let code = "to_integer(a)";
        let result = parse(&code).unwrap();
        let target = Block(vec![call!(id!("to_integer"), id!("a"))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_parenthesis() {
        let code = "to_integer a";
        let result = parse(&code).unwrap();
        let target = Block(vec![call!(id!("to_integer"), id!("a"))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_keyword_arguments_only() {
        let code = "my_function(keywords: :only)";
        let result = parse(&code).unwrap();

        let target = Block(vec![call!(
            id!("my_function"),
            list!(tuple!(atom!("keywords"), atom!("only")))
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_keyword_arguments_no_parenthesis() {
        let code = "IO.inspect label: :test";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("inspect")),
            remote_callee: Some(Box::new(atom!("IO"))),
            arguments: vec![list!(tuple!(atom!("label"), atom!("test")))],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_keyword_arguments_positional_as_well() {
        let code = "IO.inspect(my_struct, limit: :infinity)";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("inspect")),
            remote_callee: Some(Box::new(atom!("IO"))),
            arguments: vec![
                id!("my_struct"),
                list!(tuple!(atom!("limit"), atom!("infinity"))),
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_parenthesis_multiple_args() {
        let code = "to_integer a, 10, 30.2";
        let result = parse(&code).unwrap();

        let target = Block(vec![call!(
            id!("to_integer"),
            id!("a"),
            int!(10),
            float!(30.2)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_call_no_arguments() {
        let code = "to_integer()";
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Call(Call {
            target: Box::new(id!("to_integer")),
            remote_callee: None,
            arguments: vec![],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_float() {
        let code = "34.39212";
        let result = parse(&code).unwrap();

        let target = Block(vec![float!(34.39212)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_float_underscores() {
        let code = "100_000.50_000";
        let result = parse(&code).unwrap();

        let target = Block(vec![float!(100000.5)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_integer() {
        let code = "1000";
        let result = parse(&code).unwrap();

        let target = Block(vec![int!(1000)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_integer_with_underscores() {
        let code = "100_000";
        let result = parse(&code).unwrap();

        let target = Block(vec![int!(100_000)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_hex_integer() {
        let code = "0x1F";
        let result = parse(&code).unwrap();
        let target = Block(vec![int!(31)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_number() {
        let code = "0b1010";
        let result = parse(&code).unwrap();
        let target = Block(vec![int!(10)]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_octal_number() {
        let code = "0o777";
        let result = parse(&code).unwrap();
        let target = Block(vec![int!(511)]);

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
    fn parse_empty_string() {
        let code = r#""""#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::String("".to_string())]);

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

        let target = Block(vec![Expression::Module(Module {
            name: Atom("Test.CustomModule".to_string()),
            body: Box::new(Block(vec![
                Expression::FunctionDef(Function {
                    name: Identifier("func".to_string()),
                    is_private: false,
                    parameters: vec![],
                    body: Box::new(Expression::Call(Call {
                        target: Box::new(id!("priv_func")),
                        remote_callee: None,
                        arguments: vec![],
                    })),
                    guard_expression: None,
                }),
                Expression::FunctionDef(Function {
                    name: Identifier("priv_func".to_string()),
                    is_private: true,
                    parameters: vec![],
                    body: Box::new(int!(10)),
                    guard_expression: None,
                }),
            ])),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_default_parameters() {
        let code = r#"
        def func(a \\ 10) do
            10
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            parameters: vec![Parameter {
                expression: Box::new(id!("a")),
                default: Some(Box::new(int!(10))),
            }],
            body: Box::new(int!(10)),
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_no_params_no_parenthesis() {
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
            body: Box::new(Expression::Call(Call {
                target: Box::new(id!("priv_func")),
                remote_callee: None,
                arguments: vec![],
            })),
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_pattern_match() {
        let code = "
        def func(%{a: value}), do: value
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            parameters: vec![Parameter {
                expression: Box::new(Expression::Map(Map {
                    entries: vec![(atom!("a"), id!("value"))],
                })),
                default: None,
            }],
            body: Box::new(id!("value")),
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_with_guard_expression() {
        let code = "
        def guarded(a) when is_integer(a) do
            a == 10
        end
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("guarded".to_string()),
            is_private: false,
            parameters: vec![Parameter {
                expression: Box::new(id!("a")),
                default: None,
            }],
            body: Box::new(binary_operation!(id!("a"), BinaryOperator::Equal, int!(10))),
            guard_expression: Some(Box::new(call!(id!("is_integer"), id!("a")))),
        })]);

        assert_eq!(result, target);

        let code = "
        def guarded(a) when is_integer(a), do: a == 10
        ";

        let result = parse(&code).unwrap();
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
            body: Box::new(Block(vec![])),
            parameters: vec![],
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

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
            body: Box::new(Block(vec![])),
            parameters: vec![
                Parameter {
                    expression: Box::new(id!("a")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("b")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("c")),
                    default: None,
                },
            ],
            guard_expression: None,
        })]);

        assert_eq!(result, target);

        let code = "
        def func a, b, c do
        end
        ";
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_simple_keyword_definition() {
        let code = "
        def func(a, b, c), do: a + b
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionDef(Function {
            name: Identifier("func".to_string()),
            is_private: false,
            body: Box::new(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            parameters: vec![
                Parameter {
                    expression: Box::new(id!("a")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("b")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("c")),
                    default: None,
                },
            ],
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

    // TODO: add @doc to function struct as a field to show in lsp hover

    #[test]
    fn parse_list() {
        let code = "
        [a, 10, :atom]
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![list!(id!("a"), int!(10), atom!("atom"))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_list_with_keyword() {
        let code = "
        [a, b, module: :atom,   number: 200]
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![list!(
            id!("a"),
            id!("b"),
            list!(
                tuple!(atom!("module"), atom!("atom")),
                tuple!(atom!("number"), int!(200))
            )
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_list() {
        let code = "[]";
        let result = parse(&code).unwrap();
        let target = Block(vec![list!()]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_list_cons() {
        let code = "[head, 10 | _tail] = [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            list_cons!(vec![id!("head"), int!(10)], Some(Box::new(id!("_tail")))),
            BinaryOperator::Match,
            list!(int!(1), int!(2), int!(3))
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_tuple() {
        let code = "
        {a, 10, :atom}
        ";
        let result = parse(&code).unwrap();
        let target = Block(vec![tuple!(id!("a"), int!(10), atom!("atom"))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_empty_tuple() {
        let code = "{}";
        let result = parse(&code).unwrap();
        let target = Block(vec![tuple!()]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_true() {
        let code = "true";
        let result = parse(&code).unwrap();
        let target = Block(vec![bool!(true)]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_false() {
        let code = "false";
        let result = parse(&code).unwrap();
        let target = Block(vec![bool!(false)]);
        assert_eq!(result, target);
    }

    #[test]
    fn parse_nil() {
        let code = "nil";
        let result = parse(&code).unwrap();
        let target = Block(vec![nil!()]);
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
                (atom!("a"), int!(10)),
                (atom!("b"), bool!(true)),
                (atom!("c"), nil!()),
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
                (Expression::String("a".to_owned()), int!(10)),
                (Expression::String("String".to_owned()), bool!(true)),
                (Expression::String("key".to_owned()), nil!()),
                (int!(10), int!(30)),
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
                (Expression::String("a".to_owned()), int!(10)),
                (Expression::String("String".to_owned()), bool!(true)),
                (atom!("key"), int!(50)),
                (atom!("Map"), nil!()),
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
                (atom!("a"), int!(10)),
                (atom!("myweb@site.com"), bool!(true)),
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
        let target = Block(vec![Expression::AttributeDef(Attribute {
            name: Identifier("timeout".to_string()),
            value: Box::new(int!(5000)),
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
                Expression::AttributeDef(Attribute {
                    name: Identifier("moduledoc".to_string()),
                    value: Box::new(Expression::String(
                        "\n            This is a nice module!\n            ".to_string(),
                    )),
                }),
                Expression::AttributeDef(Attribute {
                    name: Identifier("doc".to_string()),
                    value: Box::new(Expression::String(
                        "\n            This is a nice function!\n            ".to_string(),
                    )),
                }),
                Expression::FunctionDef(Function {
                    name: Identifier("func".to_string()),
                    is_private: false,
                    body: Box::new(int!(10)),
                    parameters: vec![],
                    guard_expression: None,
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
                (atom!("name"), Expression::String("john".to_string())),
                (atom!("age"), int!(25)),
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
            operand: Box::new(int!(10)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_minus() {
        let code = "-1000";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::Minus,
            operand: Box::new(int!(1000)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_strict_not() {
        let code = "not true";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::StrictNot,
            operand: Box::new(bool!(true)),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unary_relaxed_not() {
        let code = "!false";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
            operator: UnaryOperator::RelaxedNot,
            operand: Box::new(bool!(false)),
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
                operand: Box::new(bool!(false)),
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
            operand: Box::new(id!("var")),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_plus() {
        let code = "20 + 40";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(20),
            BinaryOperator::Plus,
            int!(40)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_minus() {
        let code = "100 - 50";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(100),
            BinaryOperator::Minus,
            int!(50)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_star() {
        let code = "8 * 8";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(8),
            BinaryOperator::Mult,
            int!(8)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_div() {
        let code = "10 / 2";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(10),
            BinaryOperator::Div,
            int!(2)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_list_concatenation() {
        let code = "[1, 2, 3] ++ tail";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            list!(int!(1), int!(2), int!(3)),
            BinaryOperator::ListConcatenation,
            id!("tail")
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_list_subtraction() {
        let code = "cache -- [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            id!("cache"),
            BinaryOperator::ListSubtraction,
            list!(int!(1), int!(2), int!(3))
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_and() {
        let code = "true and false";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            bool!(true),
            BinaryOperator::StrictAnd,
            bool!(false)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_relaxed_and() {
        let code = ":atom && false";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            atom!("atom"),
            BinaryOperator::RelaxedAnd,
            bool!(false)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_or() {
        let code = "false or true";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            bool!(false),
            BinaryOperator::StrictOr,
            bool!(true)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_relaxed_or() {
        let code = "a || b";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            id!("a"),
            BinaryOperator::RelaxedOr,
            id!("b")
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_in() {
        let code = "1 in [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(1),
            BinaryOperator::In,
            list!(int!(1), int!(2), int!(3))
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_not_in() {
        let code = "1 not in [1, 2, 3]";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(1),
            BinaryOperator::NotIn,
            list!(int!(1), int!(2), int!(3))
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_concat() {
        let code = r#""hello" <> "world""#;
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            Expression::String("hello".to_string()),
            BinaryOperator::BinaryConcat,
            Expression::String("world".to_string())
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_pipe() {
        let code = "value |> function()";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            id!("value"),
            BinaryOperator::Pipe,
            Expression::Call(Call {
                target: Box::new(id!("function")),
                remote_callee: None,
                arguments: vec![],
            })
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_equal() {
        let code = "10 == 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(10),
            BinaryOperator::Equal,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_strict_equal() {
        let code = "10.0 === 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            float!(10.0),
            BinaryOperator::StrictEqual,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_less_than() {
        let code = "9 < 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(9),
            BinaryOperator::LessThan,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_less_than_or_equal() {
        let code = "9 <= 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(9),
            BinaryOperator::LessThanOrEqual,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_greater_than() {
        let code = "10 > 5";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(10),
            BinaryOperator::GreaterThan,
            int!(5)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_greater_than_or_equal() {
        let code = "2 >= 2";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(2),
            BinaryOperator::GreaterThanOrEqual,
            int!(2)
        )]);

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

            let target = Block(vec![binary_operation!(
                int!(2),
                BinaryOperator::Custom(operator.to_string()),
                int!(2)
            )]);

            assert_eq!(result, target);
        }
    }

    #[test]
    fn parse_simple_range() {
        let code = "0..10";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Range(Range {
            start: Box::new(int!(0)),
            end: Box::new(int!(10)),
            step: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_range_with_step() {
        let code = "0..10//2";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Range(Range {
            start: Box::new(int!(0)),
            end: Box::new(int!(10)),
            step: Some(Box::new(int!(2))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_identifier() {
        let code = "my_var = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            id!("my_var"),
            BinaryOperator::Match,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_expressions() {
        let code = "10 = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            int!(10),
            BinaryOperator::Match,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_binary_match_wildcard() {
        let code = "_ = 10";
        let result = parse(&code).unwrap();
        let target = Block(vec![binary_operation!(
            id!("_"),
            BinaryOperator::Match,
            int!(10)
        )]);

        assert_eq!(result, target);
    }

    // TODO: parse these in keyword form as well
    #[test]
    fn parse_simple_if_expression() {
        let code = r#"
        if false do
            :ok
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::If(ConditionalExpression {
            condition_expression: Box::new(bool!(false)),
            do_expression: Box::new(atom!("ok")),
            else_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_if_else_expression() {
        let code = r#"
        if a > 5 do
            10
        else
            20
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::If(ConditionalExpression {
            condition_expression: Box::new(binary_operation!(
                id!("a"),
                BinaryOperator::GreaterThan,
                int!(5)
            )),
            do_expression: Box::new(int!(10)),
            else_expression: Some(Box::new(int!(20))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_require_no_options() {
        let code = r#"
        require Logger
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Require(Require {
            target: Atom("Logger".to_string()),
            options: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_require_with_options() {
        let code = r#"
        require Logger, level: :info
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Require(Require {
            target: Atom("Logger".to_string()),
            options: Some(List {
                items: vec![tuple!(atom!("level"), atom!("info"))],
                cons: None,
            }),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_alias_no_options() {
        let code = r#"
        alias MyModule
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Alias(Alias {
            target: Atom("MyModule".to_string()),
            options: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_alias_with_options() {
        let code = r#"
        alias MyKeyword, as: Keyword
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Alias(Alias {
            target: Atom("MyKeyword".to_string()),
            options: Some(List {
                items: vec![tuple!(atom!("as"), atom!("Keyword"))],
                cons: None,
            }),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_use_no_options() {
        let code = r#"
        use MyModule
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Use(Use {
            target: Atom("MyModule".to_string()),
            options: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_use_with_options() {
        let code = r#"
        use MyModule, some: :options
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Use(Use {
            target: Atom("MyModule".to_string()),
            options: Some(List {
                items: vec![tuple!(atom!("some"), atom!("options"))],
                cons: None,
            }),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_import_no_options() {
        let code = r#"
        import String
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Import(Import {
            target: Atom("String".to_string()),
            options: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_import_with_options() {
        let code = r#"
        import String, only: [:split]
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Import(Import {
            target: Atom("String".to_string()),
            options: Some(List {
                items: vec![tuple!(atom!("only"), list!(atom!("split")))],
                cons: None,
            }),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_simple_case() {
        let code = r#"
        case a do
            10 -> true
            20 -> false
            _else -> nil
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Case(CaseExpression {
            target_expression: Box::new(id!("a")),
            arms: vec![
                CaseArm {
                    left: Box::new(int!(10)),
                    body: Box::new(bool!(true)),
                    guard_expr: None,
                },
                CaseArm {
                    left: Box::new(int!(20)),
                    body: Box::new(bool!(false)),
                    guard_expr: None,
                },
                CaseArm {
                    left: Box::new(id!("_else")),
                    body: Box::new(nil!()),
                    guard_expr: None,
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_case_with_block() {
        let code = r#"
        case a do
            10 ->
                result = 2 + 2
                result == 4
            _else ->
                nil
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Case(CaseExpression {
            target_expression: Box::new(id!("a")),
            arms: vec![
                CaseArm {
                    left: Box::new(int!(10)),
                    body: Box::new(Expression::Block(vec![
                        binary_operation!(
                            id!("result"),
                            BinaryOperator::Match,
                            binary_operation!(int!(2), BinaryOperator::Plus, int!(2))
                        ),
                        binary_operation!(id!("result"), BinaryOperator::Equal, int!(4)),
                    ])),
                    guard_expr: None,
                },
                CaseArm {
                    left: Box::new(id!("_else")),
                    body: Box::new(nil!()),
                    guard_expr: None,
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_case_with_guards() {
        let code = r#"
        case a do
            10 when true -> true
            20 when 2 == 2 -> false
            _else -> nil
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Case(CaseExpression {
            target_expression: Box::new(id!("a")),
            arms: vec![
                CaseArm {
                    left: Box::new(int!(10)),
                    body: Box::new(bool!(true)),
                    guard_expr: Some(Box::new(bool!(true))),
                },
                CaseArm {
                    left: Box::new(int!(20)),
                    body: Box::new(bool!(false)),
                    guard_expr: Some(Box::new(binary_operation!(
                        int!(2),
                        BinaryOperator::Equal,
                        int!(2)
                    ))),
                },
                CaseArm {
                    left: Box::new(id!("_else")),
                    body: Box::new(nil!()),
                    guard_expr: None,
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_cond() {
        let code = r#"
        cond do
            10 > 20 -> true
            20 < 10 -> true
            5 == 5 -> 10
            true -> false
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Cond(CondExpression {
            arms: vec![
                CondArm {
                    condition: Box::new(binary_operation!(
                        int!(10),
                        BinaryOperator::GreaterThan,
                        int!(20)
                    )),
                    body: Box::new(bool!(true)),
                },
                CondArm {
                    condition: Box::new(binary_operation!(
                        int!(20),
                        BinaryOperator::LessThan,
                        int!(10)
                    )),
                    body: Box::new(bool!(true)),
                },
                CondArm {
                    condition: Box::new(binary_operation!(int!(5), BinaryOperator::Equal, int!(5))),
                    body: Box::new(int!(10)),
                },
                CondArm {
                    condition: Box::new(bool!(true)),
                    body: Box::new(bool!(false)),
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_cond_with_block() {
        let code = r#"
        cond do
            10 > 20 ->
                var = 5
                10 > var
            true -> false
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Cond(CondExpression {
            arms: vec![
                CondArm {
                    condition: Box::new(binary_operation!(
                        int!(10),
                        BinaryOperator::GreaterThan,
                        int!(20)
                    )),
                    body: Box::new(Block(vec![
                        binary_operation!(id!("var"), BinaryOperator::Match, int!(5)),
                        binary_operation!(int!(10), BinaryOperator::GreaterThan, id!("var")),
                    ])),
                },
                CondArm {
                    condition: Box::new(bool!(true)),
                    body: Box::new(bool!(false)),
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_simple_unless() {
        let code = r#"
        unless false do
            :ok
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Unless(ConditionalExpression {
            condition_expression: Box::new(bool!(false)),
            do_expression: Box::new(atom!("ok")),
            else_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_unless_else() {
        let code = r#"
        unless false do
            :ok
        else
            :not_ok
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Unless(ConditionalExpression {
            condition_expression: Box::new(bool!(false)),
            do_expression: Box::new(atom!("ok")),
            else_expression: Some(Box::new(atom!("not_ok"))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_access_syntax() {
        let code = r#"
        my_cache[:key]
        "#;
        let result = parse(&code).unwrap();
        let target = Expression::Block(vec![Expression::Access(Access {
            target: Box::new(id!("my_cache")),
            access_expression: Box::new(atom!("key")),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_capture_expression_grouped() {
        let code = "&(&1 * &2)";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Capture(Box::new(
            binary_operation!(id!("&1"), BinaryOperator::Mult, id!("&2")),
        ))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_capture_expression_ungrouped() {
        let code = "& &1 + &2";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Capture(Box::new(
            binary_operation!(id!("&1"), BinaryOperator::Plus, id!("&2")),
        ))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_capture_function_module() {
        let code = "&Enum.map/1";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
            arity: 1,
            callee: Identifier("map".to_string()),
            remote_callee: Some(FunctionCaptureRemoteCallee::Module(Atom(
                "Enum".to_string(),
            ))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_capture_function_variable() {
        let code = "&enum.map/1";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
            arity: 1,
            callee: Identifier("map".to_string()),
            remote_callee: Some(FunctionCaptureRemoteCallee::Variable(Identifier(
                "enum".to_string(),
            ))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_function_local_capture() {
        let code = "&enum/1";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
            arity: 1,
            callee: Identifier("enum".to_string()),
            remote_callee: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_dot_access_variable() {
        let code = "my_map.my_key";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::DotAccess(DotAccess {
            body: Box::new(id!("my_map")),
            key: Identifier("my_key".to_string()),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_dot_access_literal() {
        let code = "%{a: 10}.a";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::DotAccess(DotAccess {
            body: Box::new(Expression::Map(Map {
                entries: vec![(atom!("a"), int!(10))],
            })),
            key: Identifier("a".to_string()),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_simple_lambda() {
        let code = "fn a -> a + 10 end";
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Lambda(Lambda {
            clauses: vec![LambdaClause {
                arguments: vec![id!("a")],
                body: binary_operation!(id!("a"), BinaryOperator::Plus, int!(10)),
                guard: None,
            }],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_lambda_pattern_match() {
        let code = r#"
        fn %{a: 10} -> 10 end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Lambda(Lambda {
            clauses: vec![LambdaClause {
                arguments: vec![Expression::Map(Map {
                    entries: vec![(atom!("a"), int!(10))],
                })],
                body: int!(10),
                guard: None,
            }],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_lambda_multiple_clauses() {
        let code = r#"
        fn 1, b -> 10
           _, _ -> 20
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Lambda(Lambda {
            clauses: vec![
                LambdaClause {
                    arguments: vec![int!(1), id!("b")],
                    body: int!(10),
                    guard: None,
                },
                LambdaClause {
                    arguments: vec![id!("_"), id!("_")],
                    body: int!(20),
                    guard: None,
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_lambda_body() {
        let code = r#"
        fn a ->
            result = a + 1
            result
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Lambda(Lambda {
            clauses: vec![LambdaClause {
                arguments: vec![id!("a")],
                body: Block(vec![
                    binary_operation!(
                        id!("result"),
                        BinaryOperator::Match,
                        binary_operation!(id!("a"), BinaryOperator::Plus, int!(1))
                    ),
                    id!("result"),
                ]),
                guard: None,
            }],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_lambda_guard() {
        let code = r#"
        fn a when is_integer(a) -> 10
           _ -> 20
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Lambda(Lambda {
            clauses: vec![
                LambdaClause {
                    arguments: vec![id!("a")],
                    guard: Some(Box::new(call!(id!("is_integer"), id!("a")))),
                    body: int!(10),
                },
                LambdaClause {
                    arguments: vec![id!("_")],
                    guard: None,
                    body: int!(20),
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_grouping() {
        let code = r#"
        (1 + 1)
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Grouping(Box::new(binary_operation!(
            int!(1),
            BinaryOperator::Plus,
            int!(1)
        )))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_nested_grouping() {
        let code = r#"
        ((1 + 1))
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Grouping(Box::new(Expression::Grouping(
            Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
        )))]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_quote_block() {
        let code = r#"
        quote do
            1 + 1
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Quote(Quote {
            options: None,
            body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_quote_block_options() {
        let code = r#"
        quote line: 10 do
            1 + 1
        end
        "#;
        let result = parse(&code).unwrap();

        let target = Block(vec![Expression::Quote(Quote {
            options: Some(List {
                items: vec![tuple!(atom!("line"), int!(10))],
                cons: None,
            }),
            body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
        })]);

        assert_eq!(result, target);

        let code = r#"
        quote [line: 10] do
            1 + 1
        end
        "#;
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_quote_keyword_def() {
        let code = r#"
        quote do: 1 + 1
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Quote(Quote {
            options: None,
            body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_quote_keyword_def_options() {
        let code = r#"
        quote line: 10, file: nil, do: 1 + 1
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Quote(Quote {
            options: Some(List {
                items: vec![
                    tuple!(atom!("line"), int!(10)),
                    tuple!(atom!("file"), nil!()),
                ],
                cons: None,
            }),
            body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
        })]);

        assert_eq!(result, target);

        let code = r#"
        quote [line: 10, file: nil], do: 1 + 1
        "#;
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_macro_simple_definition() {
        let code = r#"
        defmacro my_macro do

        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::MacroDef(Macro {
            name: Identifier("my_macro".to_string()),
            is_private: false,
            body: Box::new(Expression::Block(vec![])),
            guard_expression: None,
            parameters: vec![],
        })]);

        assert_eq!(result, target);

        let code = r#"
        defmacro my_macro() do

        end
        "#;
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_macro_arguments_definition() {
        let code = r#"
        defmacro my_macro(a, b, c) do
            a + b
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::MacroDef(Macro {
            name: Identifier("my_macro".to_string()),
            is_private: false,
            body: Box::new(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            guard_expression: None,
            parameters: vec![
                Parameter {
                    expression: Box::new(id!("a")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("b")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("c")),
                    default: None,
                },
            ],
        })]);

        assert_eq!(result, target);

        let code = r#"
        defmacro my_macro a, b, c do
            a + b
        end
        "#;
        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_macro_oneline() {
        let code = r#"
        defmacro my_macro(a, b, c), do: a + b
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::MacroDef(Macro {
            name: Identifier("my_macro".to_string()),
            is_private: false,
            body: Box::new(binary_operation!(id!("a"), BinaryOperator::Plus, id!("b"))),
            guard_expression: None,
            parameters: vec![
                Parameter {
                    expression: Box::new(id!("a")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("b")),
                    default: None,
                },
                Parameter {
                    expression: Box::new(id!("c")),
                    default: None,
                },
            ],
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_macro_guard() {
        let target = Block(vec![Expression::MacroDef(Macro {
            name: Identifier("my_macro".to_string()),
            is_private: false,
            body: Box::new(bool!(true)),
            guard_expression: Some(Box::new(call!(id!("is_integer"), id!("x")))),

            parameters: vec![Parameter {
                expression: Box::new(id!("x")),
                default: None,
            }],
        })]);

        let code = r#"
        defmacro my_macro(x) when is_integer(x) do
            true
        end
        "#;
        let result = parse(&code).unwrap();

        assert_eq!(result, target);

        let code = r#"
        defmacro my_macro(x) when is_integer(x), do: true
        "#;

        let result = parse(&code).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn parse_macro_default_parameters() {
        let code = r#"
        defmacrop macro(a \\ 10) do
            10
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::MacroDef(Macro {
            name: Identifier("macro".to_string()),
            is_private: true,
            parameters: vec![Parameter {
                expression: Box::new(id!("a")),
                default: Some(Box::new(int!(10))),
            }],
            body: Box::new(int!(10)),
            guard_expression: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_sigils() {
        let code = r#"
        ~s(hey_there)
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Sigil(Sigil {
            name: "s".to_string(),
            content: "hey_there".to_string(),
            modifier: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_sigils_different_separator() {
        let code = r#"
        ~r/regex^/
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Sigil(Sigil {
            name: "r".to_string(),
            content: "regex^".to_string(),
            modifier: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_longer_sigil_name() {
        let code = r#"
        ~HTML|regex^|
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Sigil(Sigil {
            name: "HTML".to_string(),
            content: "regex^".to_string(),
            modifier: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_sigil_interpolation() {
        let code = r#"
        ~HTML|with #{regex}^|
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Sigil(Sigil {
            name: "HTML".to_string(),
            content: "with #{regex}^".to_string(),
            modifier: None,
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_sigil_modifier() {
        let code = r#"
        ~r/hello/i
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![Expression::Sigil(Sigil {
            name: "r".to_string(),
            content: "hello".to_string(),
            modifier: Some("i".to_string()),
        })]);

        assert_eq!(result, target);
    }

    #[test]
    fn parse_custom_block_expressions() {
        let code = r#"
        schema "my_schema" do
            field :name, :string
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![call!(
            id!("schema"),
            string!("my_schema"),
            list!(tuple!(
                atom!("do"),
                call!(id!("field"), atom!("name"), atom!("string"))
            ))
        )]);

        assert_eq!(result, target);
    }

    // TODO: test parsing for each of the supported optional blocks

    #[test]
    fn parse_custom_block_expressions_optional_blocks() {
        let code = r#"
        schema "my_schema" do
            10
        rescue
            field :name, :string
        end
        "#;
        let result = parse(&code).unwrap();
        let target = Block(vec![call!(
            id!("schema"),
            string!("my_schema"),
            list!(
                tuple!(atom!("do"), int!(10)),
                tuple!(
                    atom!("rescue"),
                    call!(id!("field"), atom!("name"), atom!("string"))
                )
            )
        )]);

        assert_eq!(result, target);
    }

    // TODO
    // Test without any arguments
    // schema do
    //   10
    // end
}

// TODO: Target: currently parse lib/plausible_release.ex succesfully
// TODO: support `for` (tricky one)

// TODO: protocols (defprotocol, defimpl)
// Even though these should be treated as normal expressions as per the new parser ideas

// TODO: support https://hexdocs.pm/elixir/syntax-reference.html#qualified-tuples
// TODO: Support custom macros such as "schema"

// TODO: refactor how do-end are parsed and handled according to https://hexdocs.pm/elixir/syntax-reference.html#do-end-blocks
//
// TODO: parse typespecs
// parse them as string content for now
//
// TODO: macro for call structure
// TODO: separate remote from local calls in the Expression enum?

// TODO: make existing try_parse_do_block support keyword notation as well

// TODO: (fn a -> a + 1 end).(10)
// think about calls whose callee is not an identifier but rather an expression

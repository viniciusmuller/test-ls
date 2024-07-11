use crate::atom;
use core::fmt;
use std::fmt::Debug;
use std::{cell::RefCell, collections::VecDeque};
use tree_sitter::{Node, Tree};

#[derive(Debug, PartialEq, Clone)]
pub struct Float(f64);

impl Eq for Float {}

type Atom = String;
type Identifier = String;

// TODO: move is_private field to Defp variation of the expression enum
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Function {
    name: Identifier,
    is_private: bool,
    body: Box<Expression>,
    guard_expression: Option<Box<Expression>>,
    parameters: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionHead {
    name: Identifier,
    is_private: bool,
    parameters: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CustomGuard {
    name: Identifier,
    // TODO: body: Guard
    body: Box<Expression>,
    parameters: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Guard {
    left: Box<Expression>,
    right: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Macro {
    name: Identifier,
    is_private: bool,
    body: Box<Expression>,
    guard_expression: Option<Box<Expression>>,
    parameters: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    name: Atom,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct List {
    items: Vec<Expression>,
    cons: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Tuple {
    items: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Map {
    entries: Vec<(Expression, Expression)>,
    updated: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Struct {
    name: Atom,
    updated: Option<Box<Expression>>,
    entries: Vec<(Expression, Expression)>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Attribute {
    Type(Typespec),
    Opaque(Typespec),
    Typep(Typespec),
    Spec(Typespec),
    Callback(Typespec),
    Macrocallback(Typespec),
    Custom(Identifier, Box<Expression>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Call {
    target: Box<Expression>,
    remote_callee: Option<Box<Expression>>,
    do_block: Option<Do>,
    arguments: Vec<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Quote {
    options: Option<List>,
    body: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PState<'a> {
    code: String,
    expecting_do: RefCell<bool>,
    expecting_typespec: RefCell<bool>,
    tokens: Vec<Node<'a>>,
}

impl<'a> PState<'a> {
    fn init(code: &'a str, tokens: Vec<Node<'a>>) -> PState<'a> {
        PState {
            tokens,
            code: code.to_owned(),
            expecting_do: RefCell::new(true),
            expecting_typespec: RefCell::new(true),
        }
    }

    fn parse(&mut self) -> ParserResult<Expression> {
        parse_all_tokens(self, 0, Box::new(try_parse_expression))
            .map(|(expressions, offset)| (Expression::Block(expressions), offset))
    }

    fn is_eof(&self, offset: usize) -> bool {
        self.tokens.len() == offset
    }
}

pub type ParserResult<T> = Result<(T, usize), ParseError>;
pub type Parser<'a, T> = Box<dyn Fn(&'a PState<'a>, usize) -> ParserResult<T>>;

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
///
/// Since range can actually have 3 operands if taking the step into account, we don't treat
/// it as a binary operator.
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
pub struct Range {
    start: Box<Expression>,
    end: Box<Expression>,
    step: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BinaryOperation {
    operator: BinaryOperator,
    left: Box<Expression>,
    right: Box<Expression>,
}

/// Check https://hexdocs.pm/elixir/1.17.1/operators.html for more info
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum UnaryOperator {
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
pub struct UnaryOperation {
    operator: UnaryOperator,
    operand: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum OptionalBlockType {
    Else,
    Rescue,
    Catch,
    After,
    // Custom(String), custom needed?
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Do {
    body: Box<Expression>,
    optional_block: Option<OptionalBlock>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OptionalBlock {
    block: Box<Expression>,
    block_type: OptionalBlockType,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ConditionalExpression {
    condition_expression: Box<Expression>,
    do_expression: Box<Expression>,
    else_expression: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CaseExpression {
    target_expression: Box<Expression>,
    arms: Vec<Stab>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum WithClause {
    Binding(WithPattern),
    Match(WithPattern),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct WithExpression {
    do_block: Box<Expression>,
    match_clauses: Vec<WithClause>,
    else_clauses: Option<Vec<Stab>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct For {
    generators: Vec<ForGenerator>,
    block: Box<Expression>,
    uniq: Option<Box<Expression>>,
    into: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ForGenerator {
    Generator(Expression, Expression),
    Match(Expression, Expression),
    Filter(Expression),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CondExpression {
    arms: Vec<Stab>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct WithPattern {
    pattern: Expression,
    source: Expression,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Access {
    target: Box<Expression>,
    access_expression: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DotAccess {
    body: Box<Expression>,
    key: Identifier,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Alias {
    target: Vec<Atom>,
    options: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Require {
    target: Vec<Atom>,
    options: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Use {
    target: Vec<Atom>,
    options: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Import {
    target: Vec<Atom>,
    options: Option<Box<Expression>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FunctionCaptureRemoteCallee {
    Variable(Identifier),
    Module(Atom),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionCapture {
    arity: usize,
    callee: Identifier,
    remote_callee: Option<FunctionCaptureRemoteCallee>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Lambda {
    clauses: Vec<Stab>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Sigil {
    name: String,
    content: String,
    modifier: Option<String>,
}

// TODO: bitstring typespecs
//      | <<>>                          # empty bitstring
//      | <<_::size>>                   # size is 0 or a positive integer
//      | <<_::_*unit>>                 # unit is an integer from 1 to 256
//      | <<_::size, _::_*unit>>

/// https://hexdocs.pm/elixir/typespecs.html
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Typespec {
    /// my_type :: integer()
    Binding(Box<Typespec>, Box<Typespec>),
    /// (atom())
    Grouping(Box<Typespec>),
    /// integer()
    LocalType(Identifier, Vec<Typespec>),
    /// String.t()
    RemoteType(Atom, Identifier, Vec<Typespec>),
    /// :ok | :error
    Union(Box<Typespec>, Box<Typespec>),
    /// (integer(), string() -> atom())
    Lambda(Vec<Typespec>, Box<Typespec>),
    // literals
    Integer(usize),
    Atom(Atom),
    Bool(bool),
    List(Vec<Typespec>),
    // TODO: bitstring
    Nil,
    Any,
    // TODO: range
    Tuple(Vec<Typespec>),
    Map(Vec<(Typespec, Typespec)>),
    Struct(Identifier, Vec<(Typespec, Typespec)>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Stab {
    left: Vec<Expression>,
    // TODO: maybe let right be only a single Expression and wrap it with ::Block if multiple expressions
    right: Vec<Expression>,
}

/// https://hexdocs.pm/elixir/syntax-reference.html
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Bool(bool),
    Nil,
    String(String),
    Bitstring(Vec<Expression>),
    Float(Float),
    Integer(usize),
    Atom(Atom),
    Map(Map),
    Struct(Struct),
    Charlist(String),
    List(List),
    Tuple(Tuple),
    Identifier(Identifier),
    Block(Vec<Expression>),
    Module(Module),
    AttributeDef(Attribute),
    AttributeRef(Identifier),
    FunctionHead(FunctionHead),
    FunctionDef(Function),
    Defguard(CustomGuard),
    Defguardp(CustomGuard),
    Guard(Guard),
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
    With(WithExpression),
    For(For),
    Cond(CondExpression),
    Access(Access),
    DotAccess(DotAccess),
    FunctionCapture(FunctionCapture),
    Capture(Box<Expression>),
    Lambda(Lambda),
    Grouping(Box<Expression>),
    Sigil(Sigil),
    Quote(Quote),
    MacroDef(Macro),
    Stab(Stab),
    Call(Call),
    AnonymousCall(Call),
    Do(Do),
}

// TODO: Create Parser struct abstraction to hold things such as code, filename
#[derive(Debug, Clone)]
pub struct Point {
    line: usize,
    column: usize,
}

// TODO: Create Parser struct abstraction to hold things such as code, filename
#[derive(Debug, Clone)]
pub struct ParseError {
    error_type: ErrorType,
    start: Point,
    end: Point,
    offset: usize,
    file: String,
}

#[derive(Debug, Clone)]
pub enum ErrorType {
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

pub fn parse(code: &str) -> Result<Expression, ParseError> {
    let tree = get_tree(code);
    let root_node = tree.root_node();
    let mut nodes = vec![];
    flatten_node_children(root_node, &mut nodes);

    #[cfg(test)]
    dbg!(&nodes);

    let tokens = nodes.clone();

    let mut parser = PState::init(code, tokens);
    let (result, _offset) = parser.parse()?;

    Ok(result)
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

fn try_parse_module(state: &PState, offset: usize) -> Result<(Expression, usize), ParseError> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "defmodule")?;

    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (module_name, offset) = try_parse_module_name(state, offset)?;

    let (do_block, offset) = try_parse_do(state, offset)?;

    let module = Expression::Module(Module {
        name: module_name,
        body: Box::new(Expression::Do(do_block)),
    });
    Ok((module, offset))
}

fn try_parse_capture_expression_variable(
    state: &PState,
    offset: usize,
) -> ParserResult<Identifier> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "&")?;
    let (node, offset) = try_parse_grammar_name(state, offset, "integer")?;
    let content = extract_node_text(&state.code, &node);
    Ok((format!("&{}", content), offset))
}

fn try_parse_capture_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "&")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(state, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (body, offset) = try_parse_expression(state, offset)?;

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(state, offset, ")")?;
        new_offset
    } else {
        offset
    };

    let capture = Expression::Capture(Box::new(body));
    Ok((capture, offset))
}

fn try_parse_remote_function_capture(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "&")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (remote_callee, offset) = try_parse_module_name(state, offset)
        .map(|(name, offset)| (FunctionCaptureRemoteCallee::Module(name), offset))
        .or_else(|_| {
            let (name, offset) = try_parse_identifier(state, offset)?;
            Ok((FunctionCaptureRemoteCallee::Variable(name), offset))
        })?;

    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;
    let (local_callee, offset) = try_parse_identifier(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "/")?;
    let (arity, offset) = try_parse_integer(state, offset)?;

    let capture = Expression::FunctionCapture(FunctionCapture {
        arity,
        callee: local_callee,
        remote_callee: Some(remote_callee),
    });

    Ok((capture, offset))
}

fn try_parse_local_function_capture(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "&")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (local_callee, offset) = try_parse_identifier(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "/")?;
    let (arity, offset) = try_parse_integer(state, offset)?;

    let capture = Expression::FunctionCapture(FunctionCapture {
        arity,
        callee: local_callee,
        remote_callee: None,
    });

    Ok((capture, offset))
}

fn try_parse_quote(state: &PState, offset: usize) -> ParserResult<Expression> {
    // TODO: try_pares_do_block with oneline support should do the trick here for both cases
    try_parse_quote_oneline(state, offset).or_else(|_| try_parse_quote_block(state, offset))
}

fn try_parse_quote_block(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "quote")?;

    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let (options, offset) = try_parse_keyword_expressions(state, offset)
        .or_else(|_| try_parse_keyword_list(state, offset))
        .map(|(opts, offset)| (Some(opts), offset))
        .unwrap_or((None, offset));

    let (do_block, offset) = try_parse_do(state, offset)?;

    let capture = Expression::Quote(Quote {
        options,
        body: Box::new(Expression::Do(do_block)),
    });

    Ok((capture, offset))
}

fn try_parse_arguments_token<'a>(state: &'a PState, offset: usize) -> ParserResult<Node<'a>> {
    try_parse_grammar_name(state, offset, "arguments")
}

fn try_parse_call_token<'a>(state: &'a PState, offset: usize) -> ParserResult<Node<'a>> {
    try_parse_grammar_name(state, offset, "call")
}

fn try_parse_end_token<'a>(state: &'a PState, offset: usize) -> ParserResult<Node<'a>> {
    try_parse_grammar_name(state, offset, "end")
}

fn try_parse_quote_oneline(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (quote_node, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "quote")?;

    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let (mut options, offset) = try_parse_keyword_expressions(state, offset)
        .or_else(|_| try_parse_keyword_list(state, offset))
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))?;

    let (has_another_pair, offset) = try_parse_grammar_name(state, offset, ",")
        .map(|(_, offset)| (true, offset))
        .unwrap_or((false, offset));

    let (block, offset) = if has_another_pair {
        let (mut expr, offset) = try_parse_keyword_expressions(state, offset)
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
    offset: usize,
    pair: Option<(Expression, Expression)>,
    start_node: &Node,
) -> ParserResult<Expression> {
    match pair {
        Some((Expression::Atom(atom), block)) => {
            if atom == "do" {
                Ok((block, offset))
            } else {
                Err(build_unexpected_token_error(offset, start_node))
            }
        }
        Some(_) | None => Err(build_unexpected_token_error(offset, start_node)),
    }
}

fn try_parse_keyword_list(state: &PState, offset: usize) -> ParserResult<List> {
    let (_, offset) = try_parse_grammar_name(state, offset, "list")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "[")?;
    let (options, offset) = try_parse_keyword_expressions(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "]")?;
    Ok((options, offset))
}

fn try_parse_function_definition(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;

    let (offset, is_private) = match try_parse_keyword(state, offset, "defp") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(state, offset, "def")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| try_parse_grammar_name(state, offset, "call")),
    );

    if let Ok(((guard_expr, parameters, function_name), offset)) =
        try_parse_function_guard(state, offset)
    {
        let offset = try_consume(state, offset, Box::new(comma_parser));
        let (do_block, offset) = try_parse_do(state, offset)?;

        let function = Expression::FunctionDef(Function {
            name: function_name,
            body: Box::new(Expression::Do(do_block)),
            guard_expression: Some(Box::new(guard_expr)),
            parameters,
            is_private,
        });

        return Ok((function, offset));
    };

    let (function_name, offset) = try_parse_identifier(state, offset)?;

    let (parameters, offset) = try_parse_parameters(state, offset).unwrap_or((vec![], offset));

    let offset = try_consume(state, offset, Box::new(comma_parser));

    let (do_block, offset) = try_parse_do(state, offset)?;

    let function = Expression::FunctionDef(Function {
        name: function_name,
        body: Box::new(Expression::Do(do_block)),
        guard_expression: None,
        parameters,
        is_private,
    });

    Ok((function, offset))
}

fn try_parse_function_head(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;

    // TODO:: move to another function like parse_public_or_private, many places are using logic
    // similar to this one
    let (offset, is_private) = match try_parse_keyword(state, offset, "defp") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(state, offset, "def")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (function_name, offset) = try_parse_identifier(state, offset)?;

    let (parameters, offset) = try_parse_parameters(state, offset).unwrap_or((vec![], offset));

    let function = Expression::FunctionHead(FunctionHead {
        name: function_name,
        parameters,
        is_private,
    });

    Ok((function, offset))
}

fn try_parse_do_keyword(state: &PState, offset: usize) -> ParserResult<Do> {
    let (list, offset) = try_parse_keyword_expressions(state, offset)
        .or_else(|_| try_parse_keyword_list(state, offset))?;

    match keyword_fetch(&list, atom!("do")) {
        Some(body) => {
            // NOTE: Currently keyword do expressions only support optional if,
            // as the other ones apparently all exist only on blocks
            // just expand the support here if more is needed
            let optional_block =
                keyword_fetch(&list, atom!("else")).map(|else_block| OptionalBlock {
                    block: Box::new(else_block),
                    block_type: OptionalBlockType::Else,
                });

            let do_block = Do {
                body: Box::new(body),
                optional_block,
            };

            Ok((do_block, offset))
        }
        None => {
            let node = state.tokens.last().unwrap();
            return Err(build_unexpected_token_error(offset, node));
        }
    }
}

// TODO: generalize all of this function and macro definition code
fn try_parse_macro_definition(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;

    let (offset, is_private) = match try_parse_keyword(state, offset, "defmacrop") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(state, offset, "defmacro")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));
    let offset = try_consume(state, offset, Box::new(try_parse_call_token));

    if let Ok(((guard_expr, parameters, macro_name), offset)) =
        try_parse_function_guard(state, offset)
    {
        let offset = try_consume(state, offset, Box::new(comma_parser));
        let (do_block, offset) = try_parse_do(state, offset)?;

        let macro_def = Expression::MacroDef(Macro {
            name: macro_name,
            body: Box::new(Expression::Do(do_block)),
            guard_expression: Some(Box::new(guard_expr)),
            parameters,
            is_private,
        });

        return Ok((macro_def, offset));
    };

    let (macro_name, offset) = try_parse_identifier(state, offset)?;
    let (parameters, offset) = try_parse_parameters(state, offset).unwrap_or((vec![], offset));
    let offset = try_consume(state, offset, Box::new(comma_parser));
    let (do_block, offset) = try_parse_do(state, offset)?;

    let macro_def = Expression::MacroDef(Macro {
        name: macro_name,
        body: Box::new(Expression::Do(do_block)),
        guard_expression: None,
        parameters,
        is_private,
    });

    Ok((macro_def, offset))
}

fn try_parse_function_guard(
    state: &PState,
    offset: usize,
) -> ParserResult<(Expression, Vec<Expression>, Identifier)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (function_name, offset) = try_parse_identifier(state, offset)?;

    let (parameters, offset) = match try_parse_parameters(state, offset) {
        Ok(result) => result,
        Err(_) => (vec![], offset),
    };

    let (_, offset) = try_parse_grammar_name(state, offset, "when")?;
    *state.expecting_do.borrow_mut() = false;
    let (guard_expression, offset) = try_parse_expression(state, offset)?;
    *state.expecting_do.borrow_mut() = true;

    Ok(((guard_expression, parameters, function_name), offset))
}

fn try_parse_parameters(state: &PState, offset: usize) -> ParserResult<Vec<Expression>> {
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(state, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (parameters, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_parameter),
        Box::new(comma_parser),
    )
    .unwrap_or((vec![], offset));

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(state, offset, ")")?;
        new_offset
    } else {
        offset
    };

    Ok((parameters, offset))
}

fn try_parse_parameter(state: &PState, offset: usize) -> ParserResult<Expression> {
    try_parse_expression(state, offset).or_else(|_| {
        try_parse_keyword_expressions(state, offset)
            .map(|(list, offset)| (Expression::List(list), offset))
    })
}

fn try_parse_grammar_name<'a>(
    state: &'a PState,
    offset: usize,
    expected: &str,
) -> ParserResult<Node<'a>> {
    if state.is_eof(offset) {
        let node = state.tokens.last().unwrap();
        return Err(build_unexpected_token_error(offset, node));
    }

    let token = state.tokens[offset];
    let actual = token.grammar_name();

    if actual == expected {
        Ok((token, offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
    }
}

fn try_parse_identifier(state: &PState, offset: usize) -> ParserResult<Identifier> {
    let token = state.tokens[offset];
    let actual = token.grammar_name();
    let identifier_name = extract_node_text(&state.code, &token);

    if actual == "identifier" {
        // https://hexdocs.pm/elixir/syntax-reference.html#reserved-words
        let reserved = [
            "true",
            "false",
            "nil",
            "when",
            "and",
            "or",
            "not",
            "in",
            "fn",
            "do",
            "end",
            "catch",
            "rescue",
            "after",
            "else",
            // TODO: technically not reserved words, but we parse them other ways
            "...",
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

        Ok((identifier_name, offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
    }
}

fn try_parse_keyword(state: &PState, offset: usize, keyword: &str) -> ParserResult<Identifier> {
    let token = state.tokens[offset];
    let identifier_name = extract_node_text(&state.code, &token);
    let grammar_name = token.grammar_name();

    if grammar_name == "identifier" && identifier_name == keyword {
        Ok((identifier_name, offset + 1))
    } else {
        Err(build_unexpected_token_error(offset, &token))
    }
}

fn try_consume<'a, T>(state: &'a PState, offset: usize, parser: Parser<'a, T>) -> usize {
    match parser(state, offset) {
        Ok((_node, new_offset)) => new_offset,
        Err(_) => offset,
    }
}

fn try_parse_remote_call(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (remote_callee, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;

    let (local_callee, offset) = try_parse_identifier(state, offset).or_else(|_| {
        // child is the operator node
        let (_, offset) = try_parse_grammar_name(state, offset, "operator_identifier")?;
        let operator_node = state.tokens[offset];
        let operator_text = extract_node_text(&state.code, &operator_node);
        // manually increase offset for the operator text node we just extracted
        Ok((operator_text, offset + 1))
    })?;

    let (arguments, offset) = try_parse_call_arguments(state, offset)?;

    let (do_block, offset) = if *state.expecting_do.borrow() {
        try_parse_do(state, offset)
            .map(|(block, offset)| (Some(block), offset))
            .unwrap_or((None, offset))
    } else {
        (None, offset)
    };

    let call = Expression::Call(Call {
        target: Box::new(Expression::Identifier(local_callee)),
        remote_callee: Some(Box::new(remote_callee)),
        arguments,
        do_block,
    });

    Ok((call, offset))
}

// TODO: unify with try_parse_remote_call
fn try_parse_local_call(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (local_callee, offset) = try_parse_identifier(state, offset)?;

    let (arguments, offset) = try_parse_call_arguments(state, offset).unwrap_or((vec![], offset));

    let (do_block, offset) = if *state.expecting_do.borrow() {
        try_parse_do(state, offset)
            .map(|(block, offset)| (Some(block), offset))
            .unwrap_or((None, offset))
    } else {
        (None, offset)
    };

    let call = Expression::Call(Call {
        target: Box::new(Expression::Identifier(local_callee)),
        remote_callee: None,
        arguments,
        do_block,
    });

    Ok((call, offset))
}

fn try_parse_anonymous_call(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (local_callee, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;

    let (arguments, offset) = try_parse_call_arguments(state, offset).unwrap_or((vec![], offset));

    let call = Expression::AnonymousCall(Call {
        target: Box::new(local_callee),
        remote_callee: None,
        arguments,
        // NOTE: Apparently elixir-tree-sitter parser itself does not support passing do blocks in
        // anonymous calls, even though it's valid for the elixir compiler.
        // As it's a super obscure and rare thing, let's not handle it now.
        do_block: None,
    });

    Ok((call, offset))
}

fn try_parse_call_arguments(state: &PState, offset: usize) -> ParserResult<Vec<Expression>> {
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(state, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    let (mut base_arguments, offset) = match parse_expressions_sep_by_comma(state, offset) {
        Ok((expressions, new_offset)) => (expressions, new_offset),
        Err(_) => (vec![], offset),
    };

    let (keyword_arguments, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(keyword, offset)| (Some(Expression::List(keyword)), offset))
        .unwrap_or((None, offset));

    let offset = if has_parenthesis {
        let (_, new_offset) = try_parse_grammar_name(state, offset, ")")?;
        new_offset
    } else {
        offset
    };

    base_arguments.extend(keyword_arguments);
    Ok((base_arguments, offset))
}

fn comma_parser<'a>(state: &'a PState, offset: usize) -> ParserResult<Node<'a>> {
    try_parse_grammar_name(state, offset, ",")
}

// TODO: use this to parse expressions sep by comma in the function body
fn parse_expressions_sep_by_comma(state: &PState, offset: usize) -> ParserResult<Vec<Expression>> {
    try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_expression),
        Box::new(comma_parser),
    )
}

fn parse_typespecs_sep_by_comma(state: &PState, offset: usize) -> ParserResult<Vec<Typespec>> {
    try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_typespec),
        Box::new(comma_parser),
    )
}

fn try_parse_keyword_expressions(state: &PState, offset: usize) -> ParserResult<List> {
    let (_, offset) = try_parse_grammar_name(state, offset, "keywords")?;

    try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_keyword_pair),
        Box::new(comma_parser),
    )
    .map(|(pairs, offset)| {
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
    })
}

// TODO: remove all duplicated code relating expressions vs typespecs
fn try_parse_keyword_typespecs(
    state: &PState,
    offset: usize,
) -> ParserResult<Vec<(Typespec, Typespec)>> {
    let (_, offset) = try_parse_grammar_name(state, offset, "keywords")?;
    try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_keyword_typespec_pair),
        Box::new(comma_parser),
    )
}

fn try_parse_sep_by<'a, T, S>(
    state: &'a PState,
    offset: usize,
    item_parser: Parser<'a, T>,
    sep_parser: Parser<'a, S>,
) -> ParserResult<Vec<T>> {
    let (expr, offset) = item_parser(state, offset)?;
    let mut offset_mut = offset;
    let mut expressions = vec![expr];

    // TODO: Having to add these checks feel weird
    if state.is_eof(offset) {
        return Ok((expressions, offset));
    }

    let mut next_sep = sep_parser(state, offset);

    while next_sep.is_ok() {
        let (_next_comma, offset) = next_sep.unwrap();

        // Err here means trailing separator
        let (expr, offset) = match item_parser(state, offset) {
            Ok(result) => result,
            Err(_) => return Ok((expressions, offset)),
        };

        expressions.push(expr);

        if state.is_eof(offset) {
            return Ok((expressions, offset));
        }

        offset_mut = offset;
        next_sep = sep_parser(state, offset);
    }

    Ok((expressions, offset_mut))
}

fn try_parse_keyword_pair(state: &PState, offset: usize) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "pair")?;

    let (atom, offset) = match try_parse_grammar_name(state, offset, "keyword") {
        Ok((atom_node, offset)) => {
            let atom_text = extract_node_text(&state.code, &atom_node);
            // transform atom string from `atom: ` into `atom`
            let clean_atom = atom_text
                .split_whitespace()
                .collect::<String>()
                .strip_suffix(':')
                .unwrap()
                .to_string();

            (clean_atom, offset)
        }
        Err(_) => try_parse_quoted_keyword(state, offset)?,
    };

    let (value, offset) = try_parse_expression(state, offset)?;
    let pair = (Expression::Atom(atom), value);
    Ok((pair, offset))
}

fn try_parse_keyword_typespec_pair(
    state: &PState,
    offset: usize,
) -> ParserResult<(Typespec, Typespec)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "pair")?;

    let (atom, offset) = match try_parse_grammar_name(state, offset, "keyword") {
        Ok((atom_node, offset)) => {
            let atom_text = extract_node_text(&state.code, &atom_node);
            // transform atom string from `atom: ` into `atom`
            let clean_atom = atom_text
                .split_whitespace()
                .collect::<String>()
                .strip_suffix(':')
                .unwrap()
                .to_string();

            (clean_atom, offset)
        }
        Err(_) => try_parse_quoted_keyword(state, offset)?,
    };

    let (spec, offset) = try_parse_typespec(state, offset)?;
    let pair = (Typespec::Atom(atom), spec);
    Ok((pair, offset))
}

fn try_parse_quoted_keyword(state: &PState, offset: usize) -> ParserResult<Atom> {
    let (_, offset) = try_parse_grammar_name(state, offset, "quoted_keyword")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "\"")?;
    let (quoted_content, offset) = try_parse_quoted_content(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "\"")?;
    Ok((quoted_content, offset))
}

fn try_parse_cond_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "cond")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "do")?;

    let end_parser = |state, offset| try_parse_grammar_name(state, offset, "end");

    let (arms, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_stab),
        Box::new(end_parser),
    )?;

    let (_, offset) = end_parser(state, offset)?;

    let case = CondExpression { arms };

    Ok((Expression::Cond(case), offset))
}

fn try_parse_with_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "with")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    *state.expecting_do.borrow_mut() = false;
    let clause_parser = |state, offset| {
        try_parse_with_pattern(state, offset)
            .map(|(pattern, offset)| (WithClause::Match(pattern), offset))
            .or_else(|_| {
                try_parse_specific_binary_operator(state, offset, "=").map(
                    |((left, right), offset)| {
                        let with_pattern = WithPattern {
                            pattern: left,
                            source: right,
                        };

                        (WithClause::Binding(with_pattern), offset)
                    },
                )
            })
    };
    let (arguments, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(clause_parser),
        Box::new(comma_parser),
    )?;
    *state.expecting_do.borrow_mut() = true;

    let offset = try_consume(state, offset, Box::new(comma_parser));
    let ((do_block, else_block), offset) = try_parse_with_do(state, offset)?;

    let with = WithExpression {
        do_block: Box::new(do_block),
        match_clauses: arguments,
        else_clauses: else_block,
    };

    Ok((Expression::With(with), offset))
}

fn try_parse_with_pattern(state: &PState, offset: usize) -> ParserResult<WithPattern> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (pattern, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "<-")?;
    let (right, offset) = try_parse_expression(state, offset)?;

    let with_pattern = WithPattern {
        pattern,
        source: right,
    };

    Ok((with_pattern, offset))
}

fn try_parse_access_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "access_call")?;
    let (target, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "[")?;
    let (access_expression, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "]")?;

    let access = Access {
        target: Box::new(target),
        access_expression: Box::new(access_expression),
    };

    Ok((Expression::Access(access), offset))
}

fn try_parse_case_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "case")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    let (case_expression, offset) = try_parse_expression(state, offset)?;

    let (_, offset) = try_parse_grammar_name(state, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "do")?;

    let (arms, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_stab),
        Box::new(try_parse_end_token),
    )?;
    let (_, offset) = try_parse_end_token(state, offset)?;

    let case = CaseExpression {
        target_expression: Box::new(case_expression),
        arms,
    };

    Ok((Expression::Case(case), offset))
}

fn try_parse_for_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "for")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    // TODO: generators may have a guard as well
    let clause_parser = |state, offset| {
        try_parse_specific_binary_operator(state, offset, "<-")
            .map(|((left, right), offset)| (ForGenerator::Generator(left, right), offset))
            .or_else(|_| {
                try_parse_specific_binary_operator(state, offset, "=")
                    .map(|((left, right), offset)| (ForGenerator::Match(left, right), offset))
            })
            .or_else(|_| {
                try_parse_expression(state, offset)
                    .map(|(expr, offset)| (ForGenerator::Filter(expr), offset))
            })
    };

    *state.expecting_do.borrow_mut() = false;
    let (clauses, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(clause_parser),
        Box::new(comma_parser),
    )?;
    *state.expecting_do.borrow_mut() = true;

    // TODO: parse options

    let (do_block, offset) = try_parse_do(state, offset)?;

    let for_expr = For {
        generators: clauses,
        block: Box::new(Expression::Do(do_block)),
        uniq: None,
        into: None,
    };

    Ok((Expression::For(for_expr), offset))
}

fn try_parse_guard(state: &PState, offset: usize) -> ParserResult<Guard> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let (left, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "when")?;
    let (right, offset) = try_parse_expression(state, offset)?;

    let guard = Guard {
        left: Box::new(left),
        right: Box::new(right),
    };

    Ok((guard, offset))
}

fn try_parse_sigil(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "sigil")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "~")?;
    let (name_node, offset) = try_parse_grammar_name(state, offset, "sigil_name")?;
    let sigil_name = extract_node_text(&state.code, &name_node);
    let (sigil_content, offset) = try_parse_sigil_content(state, offset)?;

    let (modifier, offset) = try_parse_grammar_name(state, offset, "sigil_modifiers")
        .map(|(node, offset)| {
            let modifier = extract_node_text(&state.code, &node);
            (Some(modifier), offset)
        })
        .unwrap_or((None, offset));

    let sigil = Sigil {
        name: sigil_name,
        content: sigil_content,
        modifier,
    };

    Ok((Expression::Sigil(sigil), offset))
}

fn try_parse_sigil_content(state: &PState, offset: usize) -> ParserResult<String> {
    let sigil_open_delimiters = vec!["/", "|", "\"", "'", "(", "[", "{", "<", r#"""""#];
    let (_, offset) = try_parse_either_token(state, offset, &sigil_open_delimiters)?;

    let parser = |state, offset| {
        let (string_node, offset) = try_parse_grammar_name(state, offset, "quoted_content")
            .or_else(|_| try_parse_grammar_name(state, offset, "escape_sequence"))
            .or_else(|_| try_parse_interpolation(state, offset))?;

        let string_text = extract_node_text(&state.code, &string_node);
        Ok((string_text, offset))
    };

    let end_parser = |state, offset| {
        let sigil_close_delimiters = vec!["/", "|", "\"", "'", ")", "]", "}", ">", r#"""""#];
        try_parse_either_token(state, offset, &sigil_close_delimiters)
    };

    let (result, offset) = parse_until(state, offset, Box::new(parser), Box::new(end_parser))
        .map(|(strings, offset)| (strings.join(""), offset))?;

    let (_, offset) = end_parser(state, offset)?;
    Ok((result, offset))
}

fn try_parse_either_token<'a>(
    state: &'a PState,
    offset: usize,
    allowed_tokens: &Vec<&'static str>,
) -> ParserResult<Node<'a>> {
    for token in allowed_tokens {
        if let Ok((node, offset)) = try_parse_grammar_name(state, offset, token) {
            return Ok((node, offset));
        }
    }

    Err(build_unexpected_token_error(offset, &state.tokens[offset]))
}

fn try_parse_lambda(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "anonymous_function")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "fn")?;

    let (clauses, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_stab),
        Box::new(try_parse_end_token),
    )?;
    let (_, offset) = try_parse_grammar_name(state, offset, "end")?;

    let lambda = Lambda { clauses };

    Ok((Expression::Lambda(lambda), offset))
}

fn try_parse_stab(state: &PState, offset: usize) -> ParserResult<Stab> {
    let (_, offset) = try_parse_grammar_name(state, offset, "stab_clause")?;
    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let (arguments, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_expression),
        Box::new(comma_parser),
    )
    .unwrap_or((vec![], offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "->")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "body")?;

    let end_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "stab_clause")
            .or_else(|_| try_parse_grammar_name(state, offset, "end"))
    };

    let (body, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_expression),
        Box::new(end_parser),
    )?;

    let stab = Stab {
        left: arguments,
        right: body,
    };

    Ok((stab, offset))
}

fn try_parse_do(state: &PState, offset: usize) -> ParserResult<Do> {
    try_parse_do_block(state, offset).or_else(|_| try_parse_do_keyword(state, offset))
}

/// Parses a sugarized do block and returns a desugarized keyword structure
fn try_parse_do_block(state: &PState, offset: usize) -> ParserResult<Do> {
    let (_, offset) = try_parse_grammar_name(state, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "do")?;

    // TODO: do a bit of cleanup on this parser
    let block_end_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "end")
            .or_else(|_| try_parse_grammar_name(state, offset, "else_block"))
            .or_else(|_| try_parse_grammar_name(state, offset, "after_block"))
            .or_else(|_| try_parse_grammar_name(state, offset, "rescue_block"))
            .or_else(|_| try_parse_grammar_name(state, offset, "catch_block"))
    };

    let (mut body, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_expression),
        Box::new(block_end_parser),
    )
    .unwrap_or((vec![], offset));

    let optional_block_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "else")
            .map(|(_, offset)| (OptionalBlockType::Else, offset))
            .or_else(|_| {
                try_parse_grammar_name(state, offset, "after")
                    .map(|(_, offset)| (OptionalBlockType::After, offset))
            })
            .or_else(|_| {
                try_parse_grammar_name(state, offset, "rescue")
                    .map(|(_, offset)| (OptionalBlockType::Rescue, offset))
            })
            .or_else(|_| {
                try_parse_grammar_name(state, offset, "catch")
                    .map(|(_, offset)| (OptionalBlockType::Catch, offset))
            })
    };

    let optional_block_start_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "else_block")
            .or_else(|_| try_parse_grammar_name(state, offset, "after_block"))
            .or_else(|_| try_parse_grammar_name(state, offset, "rescue_block"))
            .or_else(|_| try_parse_grammar_name(state, offset, "catch_block"))
    };

    let (optional_block, offset) = optional_block_start_parser(state, offset)
        .and_then(|(_, offset)| {
            let (block_type, offset) = optional_block_parser(state, offset)?;

            let (mut block, offset) = parse_until(
                state,
                offset,
                Box::new(try_parse_expression),
                Box::new(try_parse_end_token),
            )?;

            let optional_block = OptionalBlock {
                block: Box::new(normalize_block(&mut block)),
                block_type,
            };

            Ok((Some(optional_block), offset))
        })
        .unwrap_or((None, offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "end")?;
    let body = normalize_block(&mut body);

    let do_block = Do {
        body: Box::new(body),
        optional_block,
    };
    Ok((do_block, offset))
}

fn normalize_block(block: &mut Vec<Expression>) -> Expression {
    if block.len() == 1 {
        block.pop().unwrap()
    } else {
        Expression::Block(block.to_owned())
    }
}

fn try_parse_with_do(
    state: &PState,
    offset: usize,
) -> ParserResult<(Expression, Option<Vec<Stab>>)> {
    try_parse_with_do_block(state, offset).or_else(|_| {
        try_parse_do_keyword(state, offset)
            .map(|(do_block, offset)| ((*do_block.body, None), offset))
    })
}

// TODO: generalize stab structure in the AST
fn try_parse_with_do_block(
    state: &PState,
    offset: usize,
) -> ParserResult<(Expression, Option<Vec<Stab>>)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "do_block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "do")?;

    // TODO: do a bit of cleanup on this parser
    let block_end_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "end")
            .or_else(|_| try_parse_grammar_name(state, offset, "else_block"))
    };

    let (mut body, offset) = parse_until(
        state,
        offset,
        Box::new(try_parse_expression),
        Box::new(block_end_parser),
    )
    .unwrap_or((vec![], offset));

    let optional_block_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "else").map(|(_, offset)| (atom!("else"), offset))
    };

    let optional_block_start_parser =
        |state, offset| try_parse_grammar_name(state, offset, "else_block");

    let (optional_block, offset) = optional_block_start_parser(state, offset)
        .and_then(|(_, offset)| {
            let (_atom, offset) = optional_block_parser(state, offset)?;

            let (else_stabs, offset) = parse_until(
                state,
                offset,
                Box::new(try_parse_stab),
                Box::new(try_parse_end_token),
            )?;

            Ok((Some(else_stabs), offset))
        })
        .unwrap_or((None, offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "end")?;
    let body = normalize_block(&mut body);

    Ok(((body, optional_block), offset))
}

fn try_parse_atom(state: &PState, offset: usize) -> ParserResult<Atom> {
    let (atom_node, offset) = try_parse_grammar_name(state, offset, "atom")?;
    let atom_string = extract_node_text(&state.code, &atom_node)[1..].to_string();
    Ok((atom_string, offset))
}

fn try_parse_quoted_atom(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "quoted_atom")?;
    let (_, offset) = try_parse_grammar_name(state, offset, ":")?;
    let delimiters = vec!["\"", "'"];
    let (_, offset) = try_parse_either_token(state, offset, &delimiters)?;
    let (atom_content, offset) = try_parse_quoted_content(state, offset)?;
    let (_, offset) = try_parse_either_token(state, offset, &delimiters)?;
    Ok((Expression::Atom(atom_content), offset))
}

fn try_parse_bool(state: &PState, offset: usize) -> ParserResult<bool> {
    let (_, offset) = try_parse_grammar_name(state, offset, "boolean")?;

    let (boolean_result, offset) = match try_parse_grammar_name(state, offset, "true") {
        Ok((_, offset)) => (true, offset),
        Err(_) => (false, offset),
    };

    let (boolean_result, offset) = if !boolean_result {
        let (_, offset) = try_parse_grammar_name(state, offset, "false")?;
        (false, offset)
    } else {
        (boolean_result, offset)
    };

    Ok((boolean_result, offset))
}

fn try_parse_nil(state: &PState, offset: usize) -> ParserResult<Expression> {
    // NOTE: nil is duplicated probably because of the flattened tree node
    let (_, offset) = try_parse_grammar_name(state, offset, "nil")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "nil")?;
    Ok((Expression::Nil, offset))
}

fn try_parse_named_typespec(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (left, offset) = try_parse_typespec(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "::")?;
    let (right, offset) = try_parse_typespec(state, offset)?;

    let spec = Typespec::Binding(Box::new(left), Box::new(right));
    Ok((spec, offset))
}

fn try_parse_type_attribute(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "@")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (node, offset) = try_parse_grammar_name(state, offset, "identifier")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (spec, offset) = try_parse_typespec(state, offset)?;

    let attribute_name = extract_node_text(&state.code, &node);
    let attribute = match attribute_name.as_str() {
        "type" => Attribute::Type(spec),
        "typep" => Attribute::Typep(spec),
        "opaque" => Attribute::Opaque(spec),
        "callback" => Attribute::Callback(spec),
        "macrocallback" => Attribute::Macrocallback(spec),
        _ => {
            let node = state.tokens.last().unwrap();
            return Err(build_unexpected_token_error(offset, node));
        }
    };

    Ok((Expression::AttributeDef(attribute), offset))
}

fn try_parse_spec_attribute(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "@")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "spec")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (spec, offset) = try_parse_typespec(state, offset)?;
    Ok((Expression::AttributeDef(Attribute::Spec(spec)), offset))
}

fn try_parse_typespec(state: &PState, offset: usize) -> ParserResult<Typespec> {
    if !*state.expecting_typespec.borrow() {
        let token = state.tokens[offset];
        return Err(build_unexpected_token_error(offset, &token));
    }

    let (body, offset) = try_parse_integer(state, offset)
        .map(|(int, offset)| (Typespec::Integer(int), offset))
        .or_else(|_| try_parse_local_type(state, offset))
        .or_else(|_| try_parse_remote_type(state, offset))
        .or_else(|_| try_parse_typespec_grouping(state, offset))
        .or_else(|_| try_parse_typespec_tuple(state, offset))
        .or_else(|_| try_parse_typespec_map(state, offset))
        .or_else(|_| try_parse_typespec_struct(state, offset))
        .or_else(|_| try_parse_typespec_list(state, offset))
        .or_else(|_| try_parse_typespec_lambda(state, offset))
        .or_else(|_| try_parse_typespec_union(state, offset))
        .or_else(|_| try_parse_wildcard_typespec(state, offset))
        .or_else(|_| {
            try_parse_bool(state, offset).map(|(bool, offset)| (Typespec::Bool(bool), offset))
        })
        .or_else(|_| {
            try_parse_identifier(state, offset)
                .map(|(target, offset)| (Typespec::LocalType(target, vec![]), offset))
        })
        .or_else(|_| {
            try_parse_atom(state, offset).map(|(atom, offset)| (Typespec::Atom(atom), offset))
        })
        .or_else(|_| {
            try_parse_module_name(state, offset)
                .map(|(atom, offset)| (Typespec::Atom(atom), offset))
        })
        .or_else(|_| try_parse_named_typespec(state, offset))
        .or_else(|_| try_parse_nil(state, offset).map(|(_, offset)| (Typespec::Nil, offset)))?;

    Ok((body, offset))
}

fn try_parse_wildcard_typespec(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "identifier")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "...")?;
    Ok((Typespec::Any, offset))
}

fn try_parse_local_type(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (callee, offset) = try_parse_identifier(state, offset)?;
    let (arguments, offset) = try_parse_typespec_call_arguments(state, offset)?;
    Ok((Typespec::LocalType(callee, arguments), offset))
}

fn try_parse_remote_type(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (remote_callee, offset) =
        try_parse_module_name(state, offset).or_else(|_| try_parse_atom(state, offset))?;
    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;
    let (local_callee, offset) = try_parse_identifier(state, offset)?;
    let (arguments, offset) =
        try_parse_typespec_call_arguments(state, offset).unwrap_or((vec![], offset));

    Ok((
        Typespec::RemoteType(remote_callee, local_callee, arguments),
        offset,
    ))
}

fn try_parse_typespec_grouping(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "(")?;
    let (spec, offset) = try_parse_typespec(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ")")?;
    let grouping = Typespec::Grouping(Box::new(spec));
    Ok((grouping, offset))
}

fn try_parse_typespec_tuple(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "tuple")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;
    let (specs, offset) = parse_typespecs_sep_by_comma(state, offset).unwrap_or((vec![], offset));
    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;

    Ok((Typespec::Tuple(specs), offset))
}

fn try_parse_typespec_lambda(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "(")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "stab_clause")?;
    let offset = try_consume(state, offset, Box::new(try_parse_arguments_token));

    let (parameters, offset) =
        parse_typespecs_sep_by_comma(state, offset).unwrap_or_else(|_| (vec![], offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "->")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "body")?;

    let (return_type, offset) = try_parse_typespec(state, offset)?;

    let (_, offset) = try_parse_grammar_name(state, offset, ")")?;

    Ok((Typespec::Lambda(parameters, Box::new(return_type)), offset))
}

fn try_parse_typespec_list(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "list")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "[")?;

    let (expressions, offset) =
        parse_typespecs_sep_by_comma(state, offset).unwrap_or_else(|_| (vec![], offset));

    // TODO: handle cons here for typespecs
    // let (expr_before_cons, cons_expr, offset) = match try_parse_list_cons(state, offset) {
    //     Ok(((expr_before_cons, cons_expr), offset)) => {
    //         (Some(expr_before_cons), Some(Box::new(cons_expr)), offset)
    //     }
    //     Err(_) => (None, None, offset),
    // };

    let (keyword, offset) =
        try_parse_keyword_typespecs(state, offset).unwrap_or_else(|_| (vec![], offset));

    let keyword = keyword
        .into_iter()
        .map(|pair| Typespec::Tuple(vec![pair.0, pair.1]));

    let (_, offset) = try_parse_grammar_name(state, offset, "]")?;

    let expressions = expressions
        .into_iter()
        // .chain(expr_before_cons)
        .chain(keyword)
        .collect();

    let list = Typespec::List(expressions);

    Ok((list, offset))
}

fn try_parse_typespec_union(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (left, offset) = try_parse_typespec(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "|")?;
    let (right, offset) = try_parse_typespec(state, offset)?;

    Ok((Typespec::Union(Box::new(left), Box::new(right)), offset))
}

fn try_parse_typespec_call_arguments(state: &PState, offset: usize) -> ParserResult<Vec<Typespec>> {
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;

    let (offset, has_parenthesis) = match try_parse_grammar_name(state, offset, "(") {
        Ok((_node, new_offset)) => (new_offset, true),
        Err(_) => (offset, false),
    };

    // Handle empty parenthesis
    if let Ok((_, offset)) = try_parse_grammar_name(state, offset, ")") {
        return Ok((vec![], offset));
    };

    let (arguments, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(try_parse_typespec),
        Box::new(comma_parser),
    )
    .unwrap_or((vec![], offset));

    let offset = if has_parenthesis {
        let (_, offset) = try_parse_grammar_name(state, offset, ")")?;
        offset
    } else {
        offset
    };

    Ok((arguments, offset))
}

// TODO: remove code repetition
fn try_parse_typespec_map(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| {
            try_parse_grammar_name(state, offset, "_items_with_trailing_separator")
        }),
    );

    let key_value_parser =
        |state, offset| try_parse_specific_binary_operator_typespec(state, offset, "=>");

    let (expression_pairs, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(key_value_parser),
        Box::new(comma_parser),
    )
    .unwrap_or_else(|_| (vec![], offset));

    // Keyword-notation-only map
    let (keyword_pairs, offset) =
        try_parse_keyword_typespecs(state, offset).unwrap_or((vec![], offset));

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;

    Ok((Typespec::Map(pairs), offset))
}

fn try_parse_typespec_struct(state: &PState, offset: usize) -> ParserResult<Typespec> {
    let (_, offset) = try_parse_grammar_name(state, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "struct")?;
    let (struct_name, offset) =
        try_parse_identifier(state, offset).or_else(|_| try_parse_module_name(state, offset))?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| {
            try_parse_grammar_name(state, offset, "_items_with_trailing_separator")
        }),
    );

    let key_value_parser =
        |state, offset| try_parse_specific_binary_operator_typespec(state, offset, "=>");

    let (expression_pairs, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(key_value_parser),
        Box::new(comma_parser),
    )
    .unwrap_or_else(|_| (vec![], offset));

    // Keyword-notation-only map
    let (keyword_pairs, offset) =
        try_parse_keyword_typespecs(state, offset).unwrap_or((vec![], offset));

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;

    Ok((Typespec::Struct(struct_name, pairs), offset))
}

fn try_parse_map(state: &PState, offset: usize) -> ParserResult<Map> {
    let (_, offset) = try_parse_grammar_name(state, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| {
            try_parse_grammar_name(state, offset, "_items_with_trailing_separator")
        }),
    );

    let (updated, offset) = try_parse_map_update(state, offset)
        .map(|(updated, offset)| (Some(Box::new(updated)), offset))
        .unwrap_or((None, offset));

    let key_value_parser = |state, offset| try_parse_specific_binary_operator(state, offset, "=>");
    let (expression_pairs, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(key_value_parser),
        Box::new(comma_parser),
    )
    .unwrap_or_else(|_| (vec![], offset));

    let (keyword_pairs, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))
        .unwrap_or((vec![], offset));

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;
    let map = Map {
        entries: pairs,
        updated,
    };

    Ok((map, offset))
}

fn try_parse_struct(state: &PState, offset: usize) -> ParserResult<Struct> {
    let (_, offset) = try_parse_grammar_name(state, offset, "map")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "%")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "struct")?;
    let (struct_name, offset) =
        try_parse_module_name(state, offset).or_else(|_| try_parse_identifier(state, offset))?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| try_parse_grammar_name(state, offset, "map_content")),
    );

    let offset = try_consume(
        state,
        offset,
        Box::new(|state, offset| {
            try_parse_grammar_name(state, offset, "_items_with_trailing_separator")
        }),
    );

    let (updated, offset) = try_parse_map_update(state, offset)
        .map(|(updated, offset)| (Some(Box::new(updated)), offset))
        .unwrap_or((None, offset));

    let key_value_parser = |state, offset| try_parse_specific_binary_operator(state, offset, "=>");
    let (expression_pairs, offset) = try_parse_sep_by(
        state,
        offset,
        Box::new(key_value_parser),
        Box::new(comma_parser),
    )
    .unwrap_or_else(|_| (vec![], offset));

    // Keyword-notation-only map
    let (keyword_pairs, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(list, offset)| (convert_keyword_expression_lists_to_tuples(list), offset))
        .unwrap_or((vec![], offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;

    let pairs = expression_pairs.into_iter().chain(keyword_pairs).collect();
    let s = Struct {
        updated,
        name: struct_name,
        entries: pairs,
    };

    Ok((s, offset))
}

fn try_parse_map_update(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (updated, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "|")?;
    Ok((updated, offset))
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

fn try_parse_grouping(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "block")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "(")?;
    let (expression, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ")")?;
    let grouping = Expression::Grouping(Box::new(expression));
    Ok((grouping, offset))
}

fn try_parse_list(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "list")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "[")?;

    // Don't allow typespec parser to consume the list cons: `last_elem | cons`
    *state.expecting_typespec.borrow_mut() = false;
    let (expressions, offset) =
        parse_expressions_sep_by_comma(state, offset).unwrap_or_else(|_| (vec![], offset));
    *state.expecting_typespec.borrow_mut() = true;

    let (expr_before_cons, cons_expr, offset) = match try_parse_list_cons(state, offset) {
        Ok(((expr_before_cons, cons_expr), offset)) => {
            (Some(expr_before_cons), Some(Box::new(cons_expr)), offset)
        }
        Err(_) => (None, None, offset),
    };

    let (keyword, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(keyword, offset)| (Some(Expression::List(keyword)), offset))
        .unwrap_or_else(|_| (None, offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "]")?;
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

fn try_parse_list_cons(state: &PState, offset: usize) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (expr_before_cons, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "|")?;
    let (cons_expr, offset) = try_parse_expression(state, offset)?;
    Ok(((expr_before_cons, cons_expr), offset))
}

fn try_parse_charlist(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "charlist")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "'")?;
    let (charlist_content, offset) = try_parse_quoted_content(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "'")?;
    Ok((Expression::Charlist(charlist_content), offset))
}

fn try_parse_char_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (char_node, offset) = try_parse_grammar_name(state, offset, "char")?;

    let char_content = extract_node_text(&state.code, &char_node)
        .strip_prefix("?")
        .unwrap_or_default()
        .to_owned();

    Ok((Expression::Charlist(char_content), offset))
}

fn try_parse_specific_binary_operator(
    state: &PState,
    offset: usize,
    expected_operator: &str,
) -> ParserResult<(Expression, Expression)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (left, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, expected_operator)?;
    let (right, offset) = try_parse_expression(state, offset)?;
    Ok(((left, right), offset))
}

fn try_parse_specific_binary_operator_typespec(
    state: &PState,
    offset: usize,
    expected_operator: &str,
) -> ParserResult<(Typespec, Typespec)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (left_spec, offset) = try_parse_typespec(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, expected_operator)?;
    let (right_spec, offset) = try_parse_typespec(state, offset)?;
    Ok(((left_spec, right_spec), offset))
}

fn try_parse_attribute_definition(state: &PState, offset: usize) -> ParserResult<Attribute> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "@")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (attribute_name, offset) = try_parse_identifier(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (value, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(list, offset)| (Expression::List(list), offset))
        .or_else(|_| try_parse_expression(state, offset))?;

    Ok((Attribute::Custom(attribute_name, Box::new(value)), offset))
}

fn try_parse_attribute_reference(state: &PState, offset: usize) -> ParserResult<Identifier> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "@")?;
    let (attribute_name, offset) = try_parse_identifier(state, offset)?;
    Ok((attribute_name, offset))
}

fn try_parse_tuple(state: &PState, offset: usize) -> Result<(Tuple, usize), ParseError> {
    let (_, offset) = try_parse_grammar_name(state, offset, "tuple")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "{")?;
    let (expressions, offset) = match parse_expressions_sep_by_comma(state, offset) {
        Ok(result) => result,
        Err(_) => (vec![], offset),
    };

    let (keyword_pairs, offset) = try_parse_keyword_expressions(state, offset)
        .map(|(list, offset)| (Some(Expression::List(list)), offset))
        .unwrap_or((None, offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;

    let expressions = expressions
        .into_iter()
        .chain(keyword_pairs)
        .collect::<Vec<_>>();
    let tuple = Tuple { items: expressions };
    Ok((tuple, offset))
}

fn try_parse_integer(state: &PState, offset: usize) -> Result<(usize, usize), ParseError> {
    let (integer_node, offset) = try_parse_grammar_name(state, offset, "integer")?;
    let integer_text = extract_node_text(&state.code, &integer_node);
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

    Ok((result, offset))
}

fn try_parse_float(state: &PState, offset: usize) -> Result<(Float, usize), ParseError> {
    let (float_node, offset) = try_parse_grammar_name(state, offset, "float")?;
    let float_text = extract_node_text(&state.code, &float_node);
    let float_text = float_text.replace('_', "");

    let f = float_text.parse::<f64>().unwrap();
    Ok((Float(f), offset))
}

fn try_parse_bitstring(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "bitstring")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "<<")?;

    // empty string
    let (body, offset) = parse_expressions_sep_by_comma(state, offset).unwrap_or((vec![], offset));

    let (_, offset) = try_parse_grammar_name(state, offset, ">>")?;

    Ok((Expression::Bitstring(body), offset))
}

fn try_parse_string(state: &PState, offset: usize) -> Result<(String, usize), ParseError> {
    let (_, offset) = try_parse_grammar_name(state, offset, "string")?;
    let quotes = vec![r#"""#, r#"""""#];
    let (_, offset) = try_parse_either_token(state, offset, &quotes)?;

    // empty string
    if let Ok((_, offset)) = try_parse_either_token(state, offset, &quotes) {
        return Ok((String::new(), offset));
    };

    let (string_text, offset) = try_parse_quoted_content(state, offset)?;
    let (_, offset) = try_parse_either_token(state, offset, &quotes)?;
    Ok((string_text, offset))
}

/// Consumes everything from a quoted_content and returns it as a concatenated string.
/// It concatenates escape sequences and interpolations into the string
fn try_parse_quoted_content(state: &PState, offset: usize) -> ParserResult<String> {
    let parser = |state, offset| {
        let (string_node, offset) = try_parse_grammar_name(state, offset, "quoted_content")
            .or_else(|_| try_parse_grammar_name(state, offset, "escape_sequence"))
            .or_else(|_| try_parse_interpolation(state, offset))?;

        let string_text = extract_node_text(&state.code, &string_node);
        Ok((string_text, offset))
    };

    let end_parser = |state, offset| {
        try_parse_grammar_name(state, offset, "\"")
            .or_else(|_| try_parse_grammar_name(state, offset, "\"\"\""))
            .or_else(|_| try_parse_grammar_name(state, offset, "'"))
    };

    parse_until(state, offset, Box::new(parser), Box::new(end_parser))
        .map(|(strings, offset)| (strings.join(""), offset))
}

// interpolations are discarded for now
// TODO: do not discard interpolations
fn try_parse_interpolation<'a>(state: &'a PState, offset: usize) -> ParserResult<Node<'a>> {
    let (interpolation_node, offset) = try_parse_grammar_name(state, offset, "interpolation")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "#{")?;
    let (_, offset) = try_parse_expression(state, offset)?;

    let (_, offset) = try_parse_grammar_name(state, offset, "}")?;
    Ok((interpolation_node, offset))
}

/// Grammar_name = alias means module names in the TS elixir grammar.
/// e.g: ThisIsAnAlias, Elixir.ThisIsAlsoAnAlias
/// as these are also technically atoms, we return them as atoms
fn try_parse_module_name(state: &PState, offset: usize) -> ParserResult<Atom> {
    let (atom_node, offset) = try_parse_grammar_name(state, offset, "alias")?;
    let atom_string = extract_node_text(&state.code, &atom_node);
    Ok((atom_string, offset))
}

fn try_parse_module_operator_name(state: &PState, offset: usize) -> ParserResult<Vec<Atom>> {
    try_parse_module_name(state, offset)
        .map(|(name, offset)| (vec![name], offset))
        .or_else(|_| try_parse_module_name_qualified_tuples(state, offset))
}

fn try_parse_module_name_qualified_tuples(
    state: &PState,
    offset: usize,
) -> ParserResult<Vec<Atom>> {
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (base_name, offset) = try_parse_module_name(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;
    let (tuple, offset) = try_parse_tuple(state, offset)?;

    let result = tuple
        .items
        .into_iter()
        .map(|item| match item {
            Expression::Atom(name) => format!("{}.{}", base_name, name),
            _else => panic!("TODO: return proper error"),
        })
        .collect();

    Ok((result, offset))
}

fn try_parse_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    try_parse_module(state, offset)
        .or_else(|err| try_parse_grouping(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_remote_call(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_dot_access(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_function_definition(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_function_head(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_macro_definition(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_if_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_unless_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_case_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_cond_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_defguard(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_with_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_for_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_access_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_lambda(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_sigil(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_quote(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_stab(state, offset)
                .map(|(stab, offset)| (Expression::Stab(stab), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_remote_function_capture(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_local_function_capture(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_capture_expression_variable(state, offset)
                .map(|(var, offset)| (Expression::Identifier(var), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_capture_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_alias(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_require(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_use(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_import(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_local_call(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_anonymous_call(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_module_name(state, offset)
                .map(|(atom, offset)| (Expression::Atom(atom), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_string(state, offset)
                .map(|(string, offset)| (Expression::String(string), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_bitstring(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_nil(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_bool(state, offset)
                .map(|(bool, offset)| (Expression::Bool(bool), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_atom(state, offset)
                .map(|(atom, offset)| (Expression::Atom(atom), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_quoted_atom(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_tuple(state, offset)
                .map(|(tuple, offset)| (Expression::Tuple(tuple), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_map(state, offset)
                .map(|(map, offset)| (Expression::Map(map), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_unary_operator(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_attribute_reference(state, offset)
                .map(|(attribute, offset)| (Expression::AttributeRef(attribute), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_charlist(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_char_expression(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_binary_operator(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_range(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_struct(state, offset)
                .map(|(s, offset)| (Expression::Struct(s), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_identifier(state, offset)
                .map(|(identifier, offset)| (Expression::Identifier(identifier), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_integer(state, offset)
                .map(|(integer, offset)| (Expression::Integer(integer), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_type_attribute(state, offset).map_err(|_| err))
        .or_else(|err| try_parse_spec_attribute(state, offset).map_err(|_| err))
        .or_else(|err| {
            try_parse_attribute_definition(state, offset)
                .map(|(attribute, offset)| (Expression::AttributeDef(attribute), offset))
                .map_err(|_| err)
        })
        .or_else(|err| {
            try_parse_guard(state, offset)
                .map(|(guard, offset)| (Expression::Guard(guard), offset))
                .map_err(|_| err)
        })
        .or_else(|err| try_parse_list(state, offset).map_err(|_| err))
        .or_else(|err| {
            // FIXME: currently we are losing the most specific parsing error somewhere in the stack
            // if you want to see the specific token that caused the error, uncomment the dbg below
            // and look at the logs
            // dbg!(&err);
            // // TODO: probably need to a dd this map_err retaining the old err to all or_else
            // calls or things of this kind
            try_parse_float(state, offset)
                .map(|(float, offset)| (Expression::Float(float), offset))
                .map_err(|_| err)
        })
}

fn try_parse_unary_operator(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "unary_operator")?;
    let (operator, offset) = try_parse_unary_operator_node(state, offset)?;
    let (operand, offset) = try_parse_expression(state, offset)?;
    let operation = Expression::UnaryOperation(UnaryOperation {
        operator,
        operand: Box::new(operand),
    });
    Ok((operation, offset))
}

fn try_parse_binary_operator(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (left, offset) = try_parse_expression(state, offset)?;
    let (operator, offset) = try_parse_binary_operator_node(state, offset)?;
    let (right, offset) = try_parse_expression(state, offset)?;
    let operation = Expression::BinaryOperation(BinaryOperation {
        operator,
        left: Box::new(left),
        right: Box::new(right),
    });
    Ok((operation, offset))
}

fn try_parse_defguard(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;

    let (offset, is_private) = try_parse_keyword(state, offset, "defguardp")
        .map(|(_, offset)| (offset, true))
        .unwrap_or((offset, false));

    let (offset, is_private) = if !is_private {
        let (_, offset) = try_parse_keyword(state, offset, "defguard")?;
        (offset, false)
    } else {
        (offset, is_private)
    };

    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (guard_name, offset) = try_parse_identifier(state, offset)?;

    let (parameters, offset) = try_parse_parameters(state, offset).unwrap_or((vec![], offset));

    let (_, offset) = try_parse_grammar_name(state, offset, "when")?;
    let (guard_body, offset) = try_parse_expression(state, offset)?;

    let guard = CustomGuard {
        name: guard_name,
        body: Box::new(guard_body),
        parameters,
    };

    let guard_expr = if is_private {
        Expression::Defguardp(guard)
    } else {
        Expression::Defguard(guard)
    };

    Ok((guard_expr, offset))
}

fn try_parse_if_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "if")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (condition_expression, offset) = try_parse_expression(state, offset)?;
    let offset = try_consume(state, offset, Box::new(comma_parser));

    let (do_block, offset) = try_parse_do(state, offset)?;
    // TODO: helper function for this
    let else_block = match do_block.optional_block {
        Some(OptionalBlock {
            block_type: OptionalBlockType::Else,
            block,
        }) => Some(block),

        Some(OptionalBlock {
            block_type: _,
            block: _,
        }) => {
            let node = state.tokens.last().unwrap();
            return Err(build_unexpected_token_error(offset, node));
        }
        None => None,
    };

    // TODO: add support back for else
    let if_expr = Expression::If(ConditionalExpression {
        condition_expression: Box::new(condition_expression),
        do_expression: do_block.body,
        else_expression: else_block,
    });

    Ok((if_expr, offset))
}

fn try_parse_unless_expression(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, "unless")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (condition_expression, offset) = try_parse_expression(state, offset)?;
    let offset = try_consume(state, offset, Box::new(comma_parser));

    let (do_block, offset) = try_parse_do(state, offset)?;
    // TODO: helper function for this
    let else_block = match do_block.optional_block {
        Some(OptionalBlock {
            block_type: OptionalBlockType::Else,
            block,
        }) => Some(block),

        Some(OptionalBlock {
            block_type: _,
            block: _,
        }) => {
            let node = state.tokens.last().unwrap();
            return Err(build_unexpected_token_error(offset, node));
        }
        None => None,
    };

    let unless_expr = Expression::Unless(ConditionalExpression {
        condition_expression: Box::new(condition_expression),
        do_expression: do_block.body,
        else_expression: else_block,
    });

    Ok((unless_expr, offset))
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

fn try_parse_require(state: &PState, offset: usize) -> ParserResult<Expression> {
    let ((required_module, options), offset) = try_parse_module_operator(state, offset, "require")?;

    let require = Expression::Require(Require {
        target: required_module,
        options,
    });

    Ok((require, offset))
}

fn try_parse_alias(state: &PState, offset: usize) -> ParserResult<Expression> {
    let ((aliased_module, options), offset) = try_parse_module_operator(state, offset, "alias")?;

    let require = Expression::Alias(Alias {
        target: aliased_module,
        options,
    });

    Ok((require, offset))
}

fn try_parse_use(state: &PState, offset: usize) -> ParserResult<Expression> {
    let ((aliased_module, options), offset) = try_parse_module_operator(state, offset, "use")?;

    let use_expr = Use {
        target: aliased_module,
        options,
    };

    Ok((Expression::Use(use_expr), offset))
}

fn try_parse_dot_access(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_grammar_name(state, offset, "dot")?;
    let (body, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, ".")?;
    let (identifier, offset) = try_parse_identifier(state, offset)?;

    let dot_access = Expression::DotAccess(DotAccess {
        body: Box::new(body),
        key: identifier,
    });

    Ok((dot_access, offset))
}

fn try_parse_import(state: &PState, offset: usize) -> ParserResult<Expression> {
    let ((imported_module, options), offset) = try_parse_module_operator(state, offset, "import")?;

    let require = Expression::Import(Import {
        target: imported_module,
        options,
    });

    Ok((require, offset))
}

/// Module operator == import, use, require, alias
fn try_parse_module_operator(
    state: &PState,
    offset: usize,
    target: &str,
) -> ParserResult<(Vec<Atom>, Option<Box<Expression>>)> {
    let (_, offset) = try_parse_grammar_name(state, offset, "call")?;
    let (_, offset) = try_parse_keyword(state, offset, target)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "arguments")?;
    let (target_module, offset) = try_parse_module_operator_name(state, offset)?;

    let (options, offset) = match try_parse_grammar_name(state, offset, ",") {
        Ok((_, offset)) => {
            let (options, offset) = try_parse_keyword_expressions(state, offset)
                .map(|(list, offset)| (Box::new(Expression::List(list)), offset))
                .or_else(|_| {
                    try_parse_expression(state, offset)
                        .map(|(expr, offset)| (Box::new(expr), offset))
                })?;

            (Some(options), offset)
        }
        Err(_) => (None, offset),
    };

    Ok(((target_module, options), offset))
}

fn try_parse_unary_operator_node(state: &PState, offset: usize) -> ParserResult<UnaryOperator> {
    let token = state.tokens[offset];
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

fn try_parse_binary_operator_node(state: &PState, offset: usize) -> ParserResult<BinaryOperator> {
    let token = state.tokens[offset];
    let operator = token.grammar_name();
    match operator {
        "=" => Ok((BinaryOperator::Match, offset + 1)),
        "+" => Ok((BinaryOperator::Plus, offset + 1)),
        "-" => Ok((BinaryOperator::Minus, offset + 1)),
        "*" => Ok((BinaryOperator::Mult, offset + 1)),
        "**" => Ok((BinaryOperator::Pow, offset + 1)),
        "::" => Ok((BinaryOperator::Type, offset + 1)),
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
        r#"\\"# => Ok((BinaryOperator::Default, offset + 1)),
        unknown_operator => try_parse_custom_operator(&token, offset, unknown_operator)
            .map(|op| (BinaryOperator::Custom(op), offset + 1)),
    }
}

fn try_parse_range(state: &PState, offset: usize) -> ParserResult<Expression> {
    let (_, offset) = try_parse_grammar_name(state, offset, "binary_operator")?;

    let (offset, has_step) = match try_parse_grammar_name(state, offset, "binary_operator") {
        Ok((_, offset)) => (offset, true),
        Err(_) => (offset, false),
    };

    let (start, offset) = try_parse_expression(state, offset)?;
    let (_, offset) = try_parse_grammar_name(state, offset, "..")?;
    let (end, offset) = try_parse_expression(state, offset)?;

    let (step, offset) = if has_step {
        let (_, offset) = try_parse_grammar_name(state, offset, "//")?;
        let (step, offset) = try_parse_expression(state, offset)?;
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
    offset: usize,
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

fn build_unexpected_token_error(offset: usize, actual_token: &Node) -> ParseError {
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

fn build_unexpected_keyword_error(offset: usize, actual_token: &Node) -> ParseError {
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
    state: &'a PState,
    offset: usize,
    parser: Parser<'a, T>,
    end_parser: Parser<'a, E>,
) -> ParserResult<Vec<T>> {
    let (expression, offset) = parser(state, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    loop {
        match end_parser(state, offset_mut) {
            // Do not consume end token, return previous offset
            Ok((_, offset)) => return Ok((expressions, offset - 1)),
            Err(_) => {
                let (expression, new_offset) = parser(state, offset_mut)?;
                expressions.push(expression);
                offset_mut = new_offset;
            }
        }
    }
}

fn parse_all_tokens<'a, T>(
    state: &'a PState,
    offset: usize,
    parser: Parser<'a, T>,
) -> ParserResult<Vec<T>> {
    if state.tokens.is_empty() {
        return Ok((vec![], 0));
    }

    let (expression, offset) = parser(state, offset)?;
    let mut expressions = vec![expression];
    let mut offset_mut = offset;

    loop {
        if state.is_eof(offset_mut) {
            return Ok((expressions, offset_mut));
        }

        let (expression, offset) = parser(state, offset_mut)?;
        expressions.push(expression);
        offset_mut = offset;
    }
}

fn extract_node_text(code: &str, node: &Node) -> String {
    code[node.byte_range()].to_string()
}

#[macro_export]
macro_rules! _do {
    ($body:expr) => {
        Expression::Do(Do {
            body: Box::new($body),
            optional_block: None,
        })
    };
}

#[macro_export]
macro_rules! atom {
    ($x:expr) => {
        Expression::Atom($x.to_string())
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

// #[macro_export]
// macro_rules! id {
//     ($x:expr) => {
//         Expression::Identifier($x.to_string())
//     };
// }

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
            arguments: vec![$($arg,)*],
            do_block: None,
        })
    };
}

#[macro_export]
macro_rules! call_no_args {
    ( $target:expr ) => {
        Expression::Call(Call {
            target: Box::new($target),
            remote_callee: None,
            arguments: vec![],
            do_block: None,
        })
    };
}

#[macro_export]
macro_rules! _type {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Type($body))
    };
}

#[macro_export]
macro_rules! typep {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Typep($body))
    };
}

#[macro_export]
macro_rules! opaque {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Opaque($body))
    };
}

#[macro_export]
macro_rules! callback {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Callback($body))
    };
}

#[macro_export]
macro_rules! macrocallback {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Macrocallback($body))
    };
}

#[macro_export]
macro_rules! spec {
    ( $body:expr ) => {
        Expression::AttributeDef(Attribute::Spec($body))
    };
}

#[macro_export]
macro_rules! with_pattern {
    ( $left:expr, $right:expr ) => {
        WithClause::Match(WithPattern {
            pattern: $left,
            source: $right,
        })
    };
}

// #[cfg(test)]
// mod tests {
//     use super::Expression::Block;
//     use super::*;

//     #[test]
//     fn parse_atom() {
//         let code = ":atom";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![atom!("atom")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_atom_with_quotes() {
//         let code = r#":"mywebsite@is_an_atom.com""#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![atom!("mywebsite@is_an_atom.com")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quoted_atom_with_interpolation() {
//         let code = ":\"john #{name}\"";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![atom!("john #{name}")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_single_quoted_atom_with_interpolation() {
//         let code = ":'john #{name}'";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![atom!("john #{name}")]);
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quoted_atom_with_escape_sequence() {
//         let code = ":\"john\n\"";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![atom!("john\n")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_alias() {
//         let code = "Elixir.MyModule";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![atom!("Elixir.MyModule")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_identifier() {
//         let code = "my_variable_name";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![id!("my_variable_name")]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_identifier_attribute() {
//         let code = "@my_module_attribute";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::AttributeRef(
//             "my_module_attribute".to_string(),
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_remote_call_module() {
//         let code = "Enum.reject(a, my_func)";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("reject")),
//             remote_callee: Some(Box::new(atom!("Enum"))),
//             arguments: vec![id!("a"), id!("my_func")],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_remote_expression_call() {
//         let code = r#":base64.encode("abc")"#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("encode")),
//             remote_callee: Some(Box::new(atom!("base64"))),
//             arguments: vec![Expression::String("abc".to_string())],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_remote_call_identifier() {
//         let code = "json_parser.parse(body)";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("parse".to_string())),
//             remote_callee: Some(Box::new(id!("json_parser"))),
//             arguments: vec![id!("body")],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_remote_call_operator() {
//         let code = "Kernel.++(list)";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("++".to_string())),
//             remote_callee: Some(Box::new(atom!("Kernel"))),
//             arguments: vec![id!("list")],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_local_call() {
//         let code = "to_integer(a)";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![call!(id!("to_integer"), id!("a"))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_no_parenthesis() {
//         let code = "to_integer a";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![call!(id!("to_integer"), id!("a"))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_lambda_no_args() {
//         let code = "
//         Repo.transaction(fn ->
//           handle_subscription_created(params)
//         end)
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("transaction")),
//             remote_callee: Some(Box::new(atom!("Repo"))),
//             arguments: vec![Expression::Lambda(Lambda {
//                 clauses: vec![Stab {
//                     left: vec![],
//                     right: vec![call!(id!("handle_subscription_created"), id!("params"))],
//                 }],
//             })],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_keyword_arguments_only() {
//         let code = "my_function(keywords: :only)";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![call!(
//             id!("my_function"),
//             list!(tuple!(atom!("keywords"), atom!("only")))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_keyword_arguments_no_parenthesis() {
//         let code = "IO.inspect label: :test";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("inspect")),
//             remote_callee: Some(Box::new(atom!("IO"))),
//             arguments: vec![list!(tuple!(atom!("label"), atom!("test")))],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_keyword_arguments_positional_as_well() {
//         let code = "IO.inspect(my_struct, limit: :infinity)";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("inspect")),
//             remote_callee: Some(Box::new(atom!("IO"))),
//             arguments: vec![
//                 id!("my_struct"),
//                 list!(tuple!(atom!("limit"), atom!("infinity"))),
//             ],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_no_parenthesis_multiple_args() {
//         let code = "to_integer a, 10, 30.2";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![call!(
//             id!("to_integer"),
//             id!("a"),
//             int!(10),
//             float!(30.2)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_no_arguments() {
//         let code = "to_integer()";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![call_no_args!(id!("to_integer"))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_float() {
//         let code = "34.39212";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![float!(34.39212)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_float_underscores() {
//         let code = "100_000.50_000";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![float!(100000.5)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_integer() {
//         let code = "1000";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![int!(1000)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_integer_with_underscores() {
//         let code = "100_000";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![int!(100_000)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_hex_integer() {
//         let code = "0x1F";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![int!(31)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_number() {
//         let code = "0b1010";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![int!(10)]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_octal_number() {
//         let code = "0o777";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![int!(511)]);

//         assert_eq!(result, target);
//     }

//     // TODO: parse integer in scientific notation
//     // 1.0e-10

//     #[test]
//     fn parse_string() {
//         let code = r#""string!""#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::String("string!".to_string())]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_string() {
//         let code = r#""""#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::String("".to_string())]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_string_escape_sequence() {
//         let code = "\"hello world!\n different\nlines\nhere\"";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::String(
//             "hello world!\n different\nlines\nhere".to_string(),
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_string_interpolation() {
//         let code = r#""this is #{string_value}!""#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::String(
//             "this is #{string_value}!".to_string(),
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_multiline_string() {
//         let code = r#"
//         """
//         Multiline!
//         String!
//         """
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::String(
//             "\n        Multiline!\n        String!\n        ".to_string(),
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_module_definition() {
//         let code = "
//         defmodule Test.CustomModule do

//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Module(Module {
//             name: "Test.CustomModule".to_string(),
//             body: Box::new(_do!(Expression::Block(vec![]))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_keyword_parameters() {
//         let code = "
//          defp do_on_ce(do: block) do
//            do_on_ee(do: nil, else: block)
//          end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "do_on_ce".to_string(),
//             is_private: true,
//             body: Box::new(_do!(call!(
//                 id!("do_on_ee"),
//                 list!(
//                     tuple!(atom!("do"), nil!()),
//                     tuple!(atom!("else"), id!("block"))
//                 )
//             ))),
//             guard_expression: None,
//             parameters: vec![list!(tuple!(atom!("do"), id!("block")))],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_multiple_modules_definition() {
//         let code = "
//         defmodule Test.CustomModule do

//         end

//         defmodule Test.SecondModule do

//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![
//             Expression::Module(Module {
//                 name: "Test.CustomModule".to_string(),
//                 body: Box::new(_do!(Expression::Block(vec![]))),
//             }),
//             Expression::Module(Module {
//                 name: "Test.SecondModule".to_string(),
//                 body: Box::new(_do!(Expression::Block(vec![]))),
//             }),
//         ]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_module_functions() {
//         let code = "
//         defmodule Test.CustomModule do
//             def func do
//                 priv_func()
//             end

//             defp priv_func do
//                 10
//             end
//         end
//         ";
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Module(Module {
//             name: "Test.CustomModule".to_string(),
//             body: Box::new(_do!(Block(vec![
//                 Expression::FunctionDef(Function {
//                     name: "func".to_string(),
//                     is_private: false,
//                     parameters: vec![],
//                     body: Box::new(_do!(call_no_args!(id!("priv_func")))),
//                     guard_expression: None,
//                 }),
//                 Expression::FunctionDef(Function {
//                     name: "priv_func".to_string(),
//                     is_private: true,
//                     parameters: vec![],
//                     body: Box::new(_do!(int!(10))),
//                     guard_expression: None,
//                 }),
//             ]))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_default_parameters() {
//         let code = r#"
//         def func(a \\ 10) do
//             10
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             parameters: vec![binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Default,
//                 int!(10)
//             )],
//             body: Box::new(_do!(int!(10))),
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_no_params_no_parenthesis() {
//         let code = "
//         def func do
//             priv_func()
//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             parameters: vec![],
//             body: Box::new(_do!(call_no_args!(id!("priv_func")))),
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_pattern_match() {
//         let code = "
//         def func(%{a: value}), do: value
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             parameters: vec![Expression::Map(Map {
//                 entries: vec![(atom!("a"), id!("value"))],
//                 updated: None,
//             })],
//             body: Box::new(_do!(id!("value"))),
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_with_guard_expression() {
//         let code = "
//         def guarded(a) when is_integer(a) do
//             a == 10
//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "guarded".to_string(),
//             is_private: false,
//             parameters: vec![id!("a")],
//             body: Box::new(_do!(binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Equal,
//                 int!(10)
//             ))),
//             guard_expression: Some(Box::new(call!(id!("is_integer"), id!("a")))),
//         })]);

//         assert_eq!(result, target);

//         let code = "
//         def guarded(a) when is_integer(a), do: a == 10
//         ";

//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_no_parameters_parenthesis() {
//         let code = "
//         def func() do
//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             body: Box::new(_do!(Block(vec![]))),
//             parameters: vec![],
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_with_parameters() {
//         let code = "
//         def func(a, b, c) do
//         end
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             body: Box::new(_do!(Block(vec![]))),
//             parameters: vec![id!("a"), id!("b"), id!("c")],
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);

//         let code = "
//         def func a, b, c do
//         end
//         ";
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_simple_keyword_definition() {
//         let code = "
//         def func(a, b, c), do: a + b
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionDef(Function {
//             name: "func".to_string(),
//             is_private: false,
//             body: Box::new(_do!(binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Plus,
//                 id!("b")
//             ))),
//             parameters: vec![id!("a"), id!("b"), id!("c")],
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     // TODO: add @doc to function struct as a field to show in lsp hover

//     #[test]
//     fn parse_list() {
//         let code = "
//         [a, 10, :atom]
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![list!(id!("a"), int!(10), atom!("atom"))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_list_with_keyword() {
//         let code = "
//         [a, b, module: :atom,   number: 200]
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![list!(
//             id!("a"),
//             id!("b"),
//             list!(
//                 tuple!(atom!("module"), atom!("atom")),
//                 tuple!(atom!("number"), int!(200))
//             )
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_list() {
//         let code = "[]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![list!()]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_list_cons() {
//         let code = "[head, 10 | _tail] = [1, 2, 3]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             list_cons!(vec![id!("head"), int!(10)], Some(Box::new(id!("_tail")))),
//             BinaryOperator::Match,
//             list!(int!(1), int!(2), int!(3))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_tuple() {
//         let code = "
//         {a, 10, :atom}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![tuple!(id!("a"), int!(10), atom!("atom"))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_tuple_keyword() {
//         let code = "
//         {:id, :binary_id, autogenerate: true}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![tuple!(
//             atom!("id"),
//             atom!("binary_id"),
//             list!(tuple!(atom!("autogenerate"), bool!(true)))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_tuple() {
//         let code = "{}";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![tuple!()]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_true() {
//         let code = "true";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![bool!(true)]);
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_false() {
//         let code = "false";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![bool!(false)]);
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_nil() {
//         let code = "nil";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![nil!()]);
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_atom_keys() {
//         let code = "
//         %{a: 10, b: true, c:    nil}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![
//                 (atom!("a"), int!(10)),
//                 (atom!("b"), bool!(true)),
//                 (atom!("c"), nil!()),
//             ],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_update_atom_key() {
//         let code = "
//         %{map | a: true}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![(atom!("a"), bool!(true))],
//             updated: Some(Box::new(id!("map"))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_update_string_key() {
//         let code = r#"
//         %{map | "string" => true}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![(string!("string"), bool!(true))],
//             updated: Some(Box::new(id!("map"))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_literal_update() {
//         let code = r#"
//         %{%{"string" => false} | "string" => true}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![(string!("string"), bool!(true))],
//             updated: Some(Box::new(Expression::Map(Map {
//                 entries: vec![(string!("string"), bool!(false))],
//                 updated: None,
//             }))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_expression_keys() {
//         let code = r#"
//         %{"a" => 10, "String" => true, "key" =>    nil, 10 => 30}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![
//                 (Expression::String("a".to_owned()), int!(10)),
//                 (Expression::String("String".to_owned()), bool!(true)),
//                 (Expression::String("key".to_owned()), nil!()),
//                 (int!(10), int!(30)),
//             ],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_mixed_keys() {
//         let code = r#"
//         %{"a" => 10, "String" => true, key: 50, Map: nil}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![
//                 (Expression::String("a".to_owned()), int!(10)),
//                 (Expression::String("String".to_owned()), bool!(true)),
//                 (atom!("key"), int!(50)),
//                 (atom!("Map"), nil!()),
//             ],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_map_quoted_atom_keys() {
//         let code = r#"
//         %{"a": 10, "myweb@site.com":   true}
//         "#;

//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![
//                 (atom!("a"), int!(10)),
//                 (atom!("myweb@site.com"), bool!(true)),
//             ],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_map() {
//         let code = "%{}";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_ignores_comments() {
//         let code = "
//         %{
//             # this comment should not be a node in our tree-sitter
//             # tree and parsing this should work fine!
//         }
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Map(Map {
//             entries: vec![],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     // TODO: test typespec specific attributes

//     #[test]
//     fn parse_attribute() {
//         let code = "@timeout 5_000";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::AttributeDef(Attribute::Custom(
//             "timeout".to_string(),
//             Box::new(int!(5000)),
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_attribute_expression() {
//         let code = "
//         @compile inline: [now: 0]
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::AttributeDef(Attribute::Custom(
//             "compile".to_string(),
//             Box::new(list!(tuple!(
//                 atom!("inline"),
//                 list!(tuple!(atom!("now"), int!(0)))
//             ))),
//         ))]);

//         // TODO: this is returning nested lists, we should update the parser state to only return
//         // the keyword tuple if inside another list
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_attributes_inside_module() {
//         let code = r#"
//         defmodule MyModule do
//             @moduledoc """
//             This is a nice module!
//             """

//             @doc """
//             This is a nice function!
//             """
//             def func() do
//                 10
//             end
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Module(Module {
//             name: "MyModule".to_string(),
//             body: Box::new(_do!(Expression::Block(vec![
//                 Expression::AttributeDef(Attribute::Custom(
//                     "moduledoc".to_string(),
//                     Box::new(Expression::String(
//                         "\n            This is a nice module!\n            ".to_string(),
//                     ))
//                 )),
//                 Expression::AttributeDef(Attribute::Custom(
//                     "doc".to_string(),
//                     Box::new(Expression::String(
//                         "\n            This is a nice function!\n            ".to_string(),
//                     ))
//                 )),
//                 Expression::FunctionDef(Function {
//                     name: "func".to_string(),
//                     is_private: false,
//                     body: Box::new(_do!(int!(10))),
//                     parameters: vec![],
//                     guard_expression: None,
//                 }),
//             ]))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_block_for_empty_input() {
//         let code = "";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![]);
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_struct() {
//         let code = r#"
//         %MyApp.User{name: "john", age: 25}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Struct(Struct {
//             name: "MyApp.User".to_string(),
//             entries: vec![
//                 (atom!("name"), Expression::String("john".to_string())),
//                 (atom!("age"), int!(25)),
//             ],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_struct_identifier_name() {
//         let code = r#"
//         %my_struct{}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Struct(Struct {
//             name: "my_struct".to_string(),
//             entries: vec![],
//             updated: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_struct_update_syntax() {
//         let code = r#"
//         %URI{existing_map | host: "host.com"}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Struct(Struct {
//             entries: vec![(atom!("host"), string!("host.com"))],
//             updated: Some(Box::new(id!("existing_map"))),
//             name: "URI".to_string(),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_unary_plus() {
//         let code = "+10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
//             operator: UnaryOperator::Plus,
//             operand: Box::new(int!(10)),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_unary_strict_not() {
//         let code = "not true";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
//             operator: UnaryOperator::StrictNot,
//             operand: Box::new(bool!(true)),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_unary_relaxed_not() {
//         let code = "!false";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
//             operator: UnaryOperator::RelaxedNot,
//             operand: Box::new(bool!(false)),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_chained_unary_operators() {
//         let code = "!!false";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
//             operator: UnaryOperator::RelaxedNot,
//             operand: Box::new(Expression::UnaryOperation(UnaryOperation {
//                 operator: UnaryOperator::RelaxedNot,
//                 operand: Box::new(bool!(false)),
//             })),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_unary_pin() {
//         let code = "^var";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::UnaryOperation(UnaryOperation {
//             operator: UnaryOperator::Pin,
//             operand: Box::new(id!("var")),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_plus() {
//         let code = "20 + 40";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(20),
//             BinaryOperator::Plus,
//             int!(40)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_associativity_plus() {
//         let code = "20 + 40 + 60";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             binary_operation!(int!(20), BinaryOperator::Plus, int!(40)),
//             BinaryOperator::Plus,
//             int!(60)
//         )]);

//         assert_eq!(result, target);
//     }

//     // TODO: handle left and right associative operators correctly
//     // #[test]
//     // fn parse_binary_associativity_mult() {
//     //     let code = "20 + 40 * 60";
//     //     let result = parse(&code).unwrap();
//     //     let target = Block(vec![binary_operation!(
//     //         int!(20),
//     //         BinaryOperator::Mult,
//     //         binary_operation!(int!(40), BinaryOperator::Plus, int!(60))
//     //     )]);

//     //     assert_eq!(result, target);
//     // }

//     #[test]
//     fn parse_binary_minus() {
//         let code = "100 - 50";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(100),
//             BinaryOperator::Minus,
//             int!(50)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_star() {
//         let code = "8 * 8";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(8),
//             BinaryOperator::Mult,
//             int!(8)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_pow() {
//         let code = "8 ** 8";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(8),
//             BinaryOperator::Pow,
//             int!(8)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_div() {
//         let code = "10 / 2";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(10),
//             BinaryOperator::Div,
//             int!(2)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_list_concatenation() {
//         let code = "[1, 2, 3] ++ tail";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             list!(int!(1), int!(2), int!(3)),
//             BinaryOperator::ListConcatenation,
//             id!("tail")
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_list_subtraction() {
//         let code = "cache -- [1, 2, 3]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("cache"),
//             BinaryOperator::ListSubtraction,
//             list!(int!(1), int!(2), int!(3))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_strict_and() {
//         let code = "true and false";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             bool!(true),
//             BinaryOperator::StrictAnd,
//             bool!(false)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_relaxed_and() {
//         let code = ":atom && false";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             atom!("atom"),
//             BinaryOperator::RelaxedAnd,
//             bool!(false)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_strict_or() {
//         let code = "false or true";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             bool!(false),
//             BinaryOperator::StrictOr,
//             bool!(true)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_relaxed_or() {
//         let code = "a || b";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("a"),
//             BinaryOperator::RelaxedOr,
//             id!("b")
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_in() {
//         let code = "1 in [1, 2, 3]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(1),
//             BinaryOperator::In,
//             list!(int!(1), int!(2), int!(3))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_not_in() {
//         let code = "1 not in [1, 2, 3]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(1),
//             BinaryOperator::NotIn,
//             list!(int!(1), int!(2), int!(3))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_concat() {
//         let code = r#""hello" <> "world""#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             Expression::String("hello".to_string()),
//             BinaryOperator::BinaryConcat,
//             Expression::String("world".to_string())
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_pipe() {
//         let code = "value |> function()";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("value"),
//             BinaryOperator::Pipe,
//             call_no_args!(id!("function"))
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_equal() {
//         let code = "10 == 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(10),
//             BinaryOperator::Equal,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_strict_equal() {
//         let code = "10.0 === 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             float!(10.0),
//             BinaryOperator::StrictEqual,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_less_than() {
//         let code = "9 < 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(9),
//             BinaryOperator::LessThan,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_less_than_or_equal() {
//         let code = "9 <= 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(9),
//             BinaryOperator::LessThanOrEqual,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_greater_than() {
//         let code = "10 > 5";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(10),
//             BinaryOperator::GreaterThan,
//             int!(5)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_greater_than_or_equal() {
//         let code = "2 >= 2";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(2),
//             BinaryOperator::GreaterThanOrEqual,
//             int!(2)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_custom_binary_operator() {
//         let custom_operators = vec![
//             "&&&", "<<<", ">>>", "<<~", "~>>", "<~", "~>", "<~>", "+++", "---",
//         ];

//         for operator in custom_operators {
//             let code = format!("2 {} 2", operator);
//             let result = parse(&code).unwrap();

//             let target = Block(vec![binary_operation!(
//                 int!(2),
//                 BinaryOperator::Custom(operator.to_string()),
//                 int!(2)
//             )]);

//             assert_eq!(result, target);
//         }
//     }

//     #[test]
//     fn parse_simple_range() {
//         let code = "0..10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Range(Range {
//             start: Box::new(int!(0)),
//             end: Box::new(int!(10)),
//             step: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_range_with_step() {
//         let code = "0..10//2";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Range(Range {
//             start: Box::new(int!(0)),
//             end: Box::new(int!(10)),
//             step: Some(Box::new(int!(2))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_match_identifier() {
//         let code = "my_var = 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("my_var"),
//             BinaryOperator::Match,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_match_expressions() {
//         let code = "10 = 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             int!(10),
//             BinaryOperator::Match,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_binary_match_wildcard() {
//         let code = "_ = 10";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("_"),
//             BinaryOperator::Match,
//             int!(10)
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_if_expression() {
//         let code = r#"
//         if false do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::If(ConditionalExpression {
//             condition_expression: Box::new(bool!(false)),
//             do_expression: Box::new(atom!("ok")),
//             else_expression: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         if false, do: :ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_if_else_expression() {
//         let code = r#"
//         if a > 5 do
//             10
//         else
//             20
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::If(ConditionalExpression {
//             condition_expression: Box::new(binary_operation!(
//                 id!("a"),
//                 BinaryOperator::GreaterThan,
//                 int!(5)
//             )),
//             do_expression: Box::new(int!(10)),
//             else_expression: Some(Box::new(int!(20))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_require_no_options() {
//         let code = r#"
//         require Logger
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Require(Require {
//             target: vec!["Logger".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_require_qualified_tuple() {
//         let code = r#"
//         require MyModule.{A, B, C}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Require(Require {
//             target: vec![
//                 "MyModule.A".to_string(),
//                 "MyModule.B".to_string(),
//                 "MyModule.C".to_string(),
//             ],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_require_with_options() {
//         let code = r#"
//         require Logger, level: :info
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Require(Require {
//             target: vec!["Logger".to_string()],
//             options: Some(Box::new(list!(tuple!(atom!("level"), atom!("info"))))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_alias_no_options() {
//         let code = r#"
//         alias MyModule
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Alias(Alias {
//             target: vec!["MyModule".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_alias_qualified_tuple() {
//         let code = r#"
//         alias MyModule.{A, B, C}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Alias(Alias {
//             target: vec![
//                 "MyModule.A".to_string(),
//                 "MyModule.B".to_string(),
//                 "MyModule.C".to_string(),
//             ],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_alias_with_options() {
//         let code = r#"
//         alias MyKeyword, as: Keyword
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Alias(Alias {
//             target: vec!["MyKeyword".to_string()],
//             options: Some(Box::new(list!(tuple!(atom!("as"), atom!("Keyword"))))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_use_no_options() {
//         let code = r#"
//         use MyModule
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Use(Use {
//             target: vec!["MyModule".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_use_qualified_tuple() {
//         let code = r#"
//         use Ecto.{Query, Repo}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Use(Use {
//             target: vec!["Ecto.Query".to_string(), "Ecto.Repo".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_use_with_options() {
//         let code = r#"
//         use MyModule, some: :options
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Use(Use {
//             target: vec!["MyModule".to_string()],
//             options: Some(Box::new(list!(tuple!(atom!("some"), atom!("options"))))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_use_literal_arg() {
//         let code = r#"
//         use MyApp.Web, :controller
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Use(Use {
//             target: vec!["MyApp.Web".to_string()],
//             options: Some(Box::new(atom!("controller"))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_import_no_options() {
//         let code = r#"
//         import String
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Import(Import {
//             target: vec!["String".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_import_qualified_tuple() {
//         let code = r#"
//         import Plug.{Conn, Middleware}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Import(Import {
//             target: vec!["Plug.Conn".to_string(), "Plug.Middleware".to_string()],
//             options: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_import_with_options() {
//         let code = r#"
//         import String, only: [:split]
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Import(Import {
//             target: vec!["String".to_string()],
//             options: Some(Box::new(list!(tuple!(
//                 atom!("only"),
//                 list!(atom!("split"))
//             )))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_case() {
//         let code = r#"
//         case a do
//             10 -> true
//             20 -> false
//             _else -> nil
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Case(CaseExpression {
//             target_expression: Box::new(id!("a")),
//             arms: vec![
//                 Stab {
//                     left: vec![int!(10)],
//                     right: vec![bool!(true)],
//                 },
//                 Stab {
//                     left: vec![int!(20)],
//                     right: vec![bool!(false)],
//                 },
//                 Stab {
//                     left: vec![id!("_else")],
//                     right: vec![nil!()],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_case_with_block() {
//         let code = r#"
//         case a do
//             10 ->
//                 result = 2 + 2
//                 result == 4
//             _else ->
//                 nil
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Case(CaseExpression {
//             target_expression: Box::new(id!("a")),
//             arms: vec![
//                 Stab {
//                     left: vec![int!(10)],
//                     right: vec![
//                         binary_operation!(
//                             id!("result"),
//                             BinaryOperator::Match,
//                             binary_operation!(int!(2), BinaryOperator::Plus, int!(2))
//                         ),
//                         binary_operation!(id!("result"), BinaryOperator::Equal, int!(4)),
//                     ],
//                 },
//                 Stab {
//                     left: vec![id!("_else")],
//                     right: vec![nil!()],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_case_with_guards() {
//         let code = r#"
//         case a do
//             10 when true -> true
//             20 when 2 == 2 -> false
//             _else -> nil
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Case(CaseExpression {
//             target_expression: Box::new(id!("a")),
//             arms: vec![
//                 Stab {
//                     left: vec![Expression::Guard(Guard {
//                         left: Box::new(int!(10)),
//                         right: Box::new(bool!(true)),
//                     })],
//                     right: vec![bool!(true)],
//                 },
//                 Stab {
//                     left: vec![Expression::Guard(Guard {
//                         left: Box::new(int!(20)),
//                         right: Box::new(binary_operation!(int!(2), BinaryOperator::Equal, int!(2))),
//                     })],
//                     right: vec![bool!(false)],
//                 },
//                 Stab {
//                     left: vec![id!("_else")],
//                     right: vec![nil!()],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_case_guard_block_clause() {
//         let code = r#"
//         case resp do
//           resp when is_integer(resp) ->
//             a = resp
//             a

//           _else ->
//             0
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Case(CaseExpression {
//             target_expression: Box::new(id!("resp")),
//             arms: vec![
//                 Stab {
//                     left: vec![Expression::Guard(Guard {
//                         left: Box::new(id!("resp")),
//                         right: Box::new(call!(id!("is_integer"), id!("resp"))),
//                     })],
//                     right: vec![
//                         binary_operation!(id!("a"), BinaryOperator::Match, id!("resp")),
//                         id!("a"),
//                     ],
//                 },
//                 Stab {
//                     left: vec![id!("_else")],
//                     right: vec![int!(0)],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_cond() {
//         let code = r#"
//         cond do
//             10 > 20 -> true
//             20 < 10 -> true
//             5 == 5 -> 10
//             true -> false
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Cond(CondExpression {
//             arms: vec![
//                 Stab {
//                     left: vec![binary_operation!(
//                         int!(10),
//                         BinaryOperator::GreaterThan,
//                         int!(20)
//                     )],
//                     right: vec![bool!(true)],
//                 },
//                 Stab {
//                     left: vec![binary_operation!(
//                         int!(20),
//                         BinaryOperator::LessThan,
//                         int!(10)
//                     )],
//                     right: vec![bool!(true)],
//                 },
//                 Stab {
//                     left: vec![binary_operation!(int!(5), BinaryOperator::Equal, int!(5))],
//                     right: vec![int!(10)],
//                 },
//                 Stab {
//                     left: vec![bool!(true)],
//                     right: vec![bool!(false)],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_cond_with_block() {
//         let code = r#"
//         cond do
//             10 > 20 ->
//                 var = 5
//                 10 > var
//             true -> false
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Cond(CondExpression {
//             arms: vec![
//                 Stab {
//                     left: vec![binary_operation!(
//                         int!(10),
//                         BinaryOperator::GreaterThan,
//                         int!(20)
//                     )],
//                     right: vec![
//                         binary_operation!(id!("var"), BinaryOperator::Match, int!(5)),
//                         binary_operation!(int!(10), BinaryOperator::GreaterThan, id!("var")),
//                     ],
//                 },
//                 Stab {
//                     left: vec![bool!(true)],
//                     right: vec![bool!(false)],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_unless() {
//         let code = r#"
//         unless false do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Unless(ConditionalExpression {
//             condition_expression: Box::new(bool!(false)),
//             do_expression: Box::new(atom!("ok")),
//             else_expression: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         unless false, do: :ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_unless_else() {
//         let code = r#"
//         unless false do
//             :ok
//         else
//             :not_ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Unless(ConditionalExpression {
//             condition_expression: Box::new(bool!(false)),
//             do_expression: Box::new(atom!("ok")),
//             else_expression: Some(Box::new(atom!("not_ok"))),
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         unless false,
//             do: :ok,
//             else: :not_ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_access_syntax() {
//         let code = r#"
//         my_cache[:key]
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Expression::Block(vec![Expression::Access(Access {
//             target: Box::new(id!("my_cache")),
//             access_expression: Box::new(atom!("key")),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_capture_expression_grouped() {
//         let code = "&(&1 * &2)";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Capture(Box::new(binary_operation!(
//             id!("&1"),
//             BinaryOperator::Mult,
//             id!("&2")
//         )))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_capture_expression_ungrouped() {
//         let code = "& &1 + &2";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Capture(Box::new(binary_operation!(
//             id!("&1"),
//             BinaryOperator::Plus,
//             id!("&2")
//         )))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_capture_function_module() {
//         let code = "&Enum.map/1";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
//             arity: 1,
//             callee: "map".to_string(),
//             remote_callee: Some(FunctionCaptureRemoteCallee::Module("Enum".to_string())),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_capture_function_variable() {
//         let code = "&enum.map/1";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
//             arity: 1,
//             callee: "map".to_string(),
//             remote_callee: Some(FunctionCaptureRemoteCallee::Variable("enum".to_string())),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_local_capture() {
//         let code = "&enum/1";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionCapture(FunctionCapture {
//             arity: 1,
//             callee: "enum".to_string(),
//             remote_callee: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_dot_access_variable() {
//         let code = "my_map.my_key";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::DotAccess(DotAccess {
//             body: Box::new(id!("my_map")),
//             key: "my_key".to_string(),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_dot_access_literal() {
//         let code = "%{a: 10}.a";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::DotAccess(DotAccess {
//             body: Box::new(Expression::Map(Map {
//                 entries: vec![(atom!("a"), int!(10))],
//                 updated: None,
//             })),
//             key: "a".to_string(),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_lambda() {
//         let code = "fn a -> a + 10 end";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Lambda(Lambda {
//             clauses: vec![Stab {
//                 left: vec![id!("a")],
//                 right: vec![binary_operation!(id!("a"), BinaryOperator::Plus, int!(10))],
//             }],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_pattern_match() {
//         let code = r#"
//         fn %{a: 10} -> 10 end
//         "#;
//         let result = parse(&code).unwrap();
//         let vec = vec![Expression::Lambda(Lambda {
//             clauses: vec![Stab {
//                 left: vec![Expression::Map(Map {
//                     entries: vec![(atom!("a"), int!(10))],
//                     updated: None,
//                 })],
//                 right: vec![int!(10)],
//             }],
//         })];
//         let target = Block(vec);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_multiple_clauses() {
//         let code = r#"
//         fn 1, b -> 10
//            _, _ -> 20
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Lambda(Lambda {
//             clauses: vec![
//                 Stab {
//                     left: vec![int!(1), id!("b")],
//                     right: vec![int!(10)],
//                 },
//                 Stab {
//                     left: vec![id!("_"), id!("_")],
//                     right: vec![int!(20)],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_body() {
//         let code = r#"
//         fn a ->
//             result = a + 1
//             result
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Lambda(Lambda {
//             clauses: vec![Stab {
//                 left: vec![id!("a")],
//                 right: vec![
//                     binary_operation!(
//                         id!("result"),
//                         BinaryOperator::Match,
//                         binary_operation!(id!("a"), BinaryOperator::Plus, int!(1))
//                     ),
//                     id!("result"),
//                 ],
//             }],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_guard() {
//         let code = r#"
//         fn a when is_integer(a) -> 10
//            _ -> 20
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Lambda(Lambda {
//             clauses: vec![
//                 Stab {
//                     left: vec![Expression::Guard(Guard {
//                         left: Box::new(id!("a")),
//                         right: Box::new(call!(id!("is_integer"), id!("a"))),
//                     })],
//                     right: vec![int!(10)],
//                 },
//                 Stab {
//                     left: vec![id!("_")],
//                     right: vec![int!(20)],
//                 },
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_grouping() {
//         let code = r#"
//         (1 + 1)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Grouping(Box::new(binary_operation!(
//             int!(1),
//             BinaryOperator::Plus,
//             int!(1)
//         )))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_nested_grouping() {
//         let code = r#"
//         ((1 + 1))
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Grouping(Box::new(Expression::Grouping(
//             Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
//         )))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quote_block() {
//         let code = r#"
//         quote do
//             1 + 1
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(_do!(binary_operation!(
//                 int!(1),
//                 BinaryOperator::Plus,
//                 int!(1)
//             ))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quote_block_options() {
//         let code = r#"
//         quote line: 10 do
//             1 + 1
//         end
//         "#;
//         let result = parse(&code).unwrap();

//         let target = Block(vec![Expression::Quote(Quote {
//             options: Some(List {
//                 items: vec![tuple!(atom!("line"), int!(10))],
//                 cons: None,
//             }),
//             body: Box::new(_do!(binary_operation!(
//                 int!(1),
//                 BinaryOperator::Plus,
//                 int!(1)
//             ))),
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         quote [line: 10] do
//             1 + 1
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quote_keyword_def() {
//         let code = r#"
//         quote do: 1 + 1
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_quote_keyword_def_options() {
//         let code = r#"
//         quote line: 10, file: nil, do: 1 + 1
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: Some(List {
//                 items: vec![
//                     tuple!(atom!("line"), int!(10)),
//                     tuple!(atom!("file"), nil!()),
//                 ],
//                 cons: None,
//             }),
//             body: Box::new(binary_operation!(int!(1), BinaryOperator::Plus, int!(1))),
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         quote [line: 10, file: nil], do: 1 + 1
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macro_simple_definition() {
//         let code = r#"
//         defmacro my_macro do

//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::MacroDef(Macro {
//             name: "my_macro".to_string(),
//             is_private: false,
//             body: Box::new(_do!(Expression::Block(vec![]))),
//             guard_expression: None,
//             parameters: vec![],
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         defmacro my_macro() do

//         end
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macro_arguments_definition() {
//         let code = r#"
//         defmacro my_macro(a, b, c) do
//             a + b
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::MacroDef(Macro {
//             name: "my_macro".to_string(),
//             is_private: false,
//             body: Box::new(_do!(binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Plus,
//                 id!("b")
//             ))),
//             guard_expression: None,
//             parameters: vec![id!("a"), id!("b"), id!("c")],
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         defmacro my_macro a, b, c do
//             a + b
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macro_oneline() {
//         let code = r#"
//         defmacro my_macro(a, b, c), do: a + b
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::MacroDef(Macro {
//             name: "my_macro".to_string(),
//             is_private: false,
//             body: Box::new(_do!(binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Plus,
//                 id!("b")
//             ))),
//             guard_expression: None,
//             parameters: vec![id!("a"), id!("b"), id!("c")],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macro_guard() {
//         let target = Block(vec![Expression::MacroDef(Macro {
//             name: "my_macro".to_string(),
//             is_private: false,
//             body: Box::new(_do!(bool!(true))),
//             guard_expression: Some(Box::new(call!(id!("is_integer"), id!("x")))),
//             parameters: vec![id!("x")],
//         })]);

//         let code = r#"
//         defmacro my_macro(x) when is_integer(x) do
//             true
//         end
//         "#;
//         let result = parse(&code).unwrap();

//         assert_eq!(result, target);

//         let code = r#"
//         defmacro my_macro(x) when is_integer(x), do: true
//         "#;

//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macro_default_parameters() {
//         let code = r#"
//         defmacrop macro(a \\ 10) do
//             10
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::MacroDef(Macro {
//             name: "macro".to_string(),
//             is_private: true,
//             parameters: vec![binary_operation!(
//                 id!("a"),
//                 BinaryOperator::Default,
//                 int!(10)
//             )],
//             body: Box::new(_do!(int!(10))),
//             guard_expression: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigils() {
//         let code = r#"
//         ~s(hey_there)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "s".to_string(),
//             content: "hey_there".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigils_different_separator() {
//         let code = r#"
//         ~r/regex^/
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "r".to_string(),
//             content: "regex^".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_longer_sigil_name() {
//         let code = r#"
//         ~HTML|regex^|
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "HTML".to_string(),
//             content: "regex^".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigil_interpolation() {
//         let code = r#"
//         ~HTML|with #{regex}^|
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "HTML".to_string(),
//             content: "with #{regex}^".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigil_regex() {
//         let code = r#"
//         ~r/^\/.*/
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "r".to_string(),
//             content: r#"^\/.*"#.to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigil_modifier() {
//         let code = r#"
//         ~r/hello/i
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "r".to_string(),
//             content: "hello".to_string(),
//             modifier: Some("i".to_string()),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigil_multiline_delimiter() {
//         let code = r#"
//         ~H"""
//         <html><%= hey there! %></html>
//         """
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "H".to_string(),
//             content: "\n        <html><%= hey there! %></html>\n        ".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_sigil_square_delimiter() {
//         let code = "~D[2135-01-01]";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Sigil(Sigil {
//             name: "D".to_string(),
//             content: "2135-01-01".to_string(),
//             modifier: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_custom_block_expressions() {
//         let code = r#"
//         schema "my_schema" do
//             field :name, :string
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("schema")),
//             remote_callee: None,
//             arguments: vec![string!("my_schema")],
//             do_block: Some(Do {
//                 body: Box::new(call!(id!("field"), atom!("name"), atom!("string"))),
//                 optional_block: None,
//             }),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_custom_block_expressions_optional_blocks() {
//         let code = r#"
//         schema "my_schema" do
//             10
//         rescue
//             field :name, :string
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("schema")),
//             remote_callee: None,
//             arguments: vec![string!("my_schema")],
//             do_block: Some(Do {
//                 body: Box::new(int!(10)),
//                 optional_block: Some(OptionalBlock {
//                     block: Box::new(call!(id!("field"), atom!("name"), atom!("string"))),
//                     block_type: OptionalBlockType::Rescue,
//                 }),
//             }),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_custom_block_expressions_no_args() {
//         let code = r#"
//         my_op do
//             10
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("my_op")),
//             remote_callee: None,
//             arguments: vec![],
//             do_block: Some(Do {
//                 body: Box::new(int!(10)),
//                 optional_block: None,
//             }),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_integer_literal_typespec() {
//         let code = r#"
//         @type my_type :: 10
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("my_type".to_string(), vec![])),
//             Box::new(Typespec::Integer(10))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_atom_literal_typespec() {
//         let code = r#"
//         @type type :: :api_key
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::Atom("api_key".to_string()))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_atom_module_literal_typespec() {
//         let code = r#"
//         @type type :: MyBehaviour
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::Atom("MyBehaviour".to_string()))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_bool_literal_typespec() {
//         let code = r#"
//         @type truth :: true
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("truth".to_string(), vec![])),
//             Box::new(Typespec::Bool(true))
//         ))]);

//         assert_eq!(result, target);

//         let code = r#"
//         @type falsy :: false
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("falsy".to_string(), vec![])),
//             Box::new(Typespec::Bool(false))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_grouping_typespec() {
//         let code = r#"
//         @type truth :: (yes :: true)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("truth".to_string(), vec![])),
//             Box::new(Typespec::Grouping(Box::new(Typespec::Binding(
//                 Box::new(Typespec::LocalType("yes".to_string(), vec![])),
//                 Box::new(Typespec::Bool(true))
//             ))))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_associativity_typespec() {
//         let code = r#"
//         @type truth :: yes :: true
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("truth".to_string(), vec![])),
//             Box::new(Typespec::Binding(
//                 Box::new(Typespec::LocalType("yes".to_string(), vec![])),
//                 Box::new(Typespec::Bool(true))
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_nil_literal_typespec() {
//         let code = r#"
//         @type nothing :: nil
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("nothing".to_string(), vec![])),
//             Box::new(Typespec::Nil)
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_call_typespec() {
//         let code = r#"
//         @type numberino :: integer()
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("numberino".to_string(), vec![])),
//             Box::new(Typespec::LocalType("integer".to_string(), vec![])),
//         ))]);

//         assert_eq!(result, target);

//         let code = r#"
//         @type numberino :: integer
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_parameterized_call_typespec() {
//         let code = r#"
//         @type numbers :: list(integer())
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("numbers".to_string(), vec![])),
//             Box::new(Typespec::LocalType(
//                 "list".to_string(),
//                 vec![Typespec::LocalType("integer".to_string(), vec![])]
//             )),
//         ))]);

//         assert_eq!(result, target);

//         let code = r#"
//         @type numbers :: list integer
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_tuple_typespec() {
//         let code = r#"
//         @type elems :: {integer(), string()}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("elems".to_string(), vec![])),
//             Box::new(Typespec::Tuple(vec![
//                 Typespec::LocalType("integer".to_string(), vec![]),
//                 Typespec::LocalType("string".to_string(), vec![]),
//             ]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_tuple_empty_typespec() {
//         let code = r#"
//         @type elems :: {}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("elems".to_string(), vec![])),
//             Box::new(Typespec::Tuple(vec![]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_union_typespec() {
//         let code = r#"
//         @type union :: :a | :b
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("union".to_string(), vec![])),
//             Box::new(Typespec::Union(
//                 Box::new(Typespec::Atom("a".to_string())),
//                 Box::new(Typespec::Atom("b".to_string())),
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_nested_union_typespec() {
//         let code = r#"
//         @type union :: :a | :b | :c
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("union".to_string(), vec![])),
//             Box::new(Typespec::Union(
//                 Box::new(Typespec::Atom("a".to_string())),
//                 Box::new(Typespec::Union(
//                     Box::new(Typespec::Atom("b".to_string())),
//                     Box::new(Typespec::Atom("c".to_string()))
//                 ))
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_any() {
//         let code = r#"
//         @type ellipsis :: ...
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("ellipsis".to_string(), vec![])),
//             Box::new(Typespec::Any)
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_list() {
//         let code = r#"
//         @type type :: [integer()]
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::List(vec![Typespec::LocalType(
//                 "integer".to_string(),
//                 vec![]
//             )]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_list_empty() {
//         let code = r#"
//         @type type :: []
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::List(vec![]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_list_keyword() {
//         let code = r#"
//         @type type :: [key: integer]
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::List(vec![Typespec::Tuple(vec![
//                 Typespec::Atom("key".to_string()),
//                 Typespec::LocalType("integer".to_string(), vec![])
//             ])]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_non_empty_of() {
//         let code = r#"
//         @type type :: [node(), ...]
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::List(vec![
//                 Typespec::LocalType("node".to_string(), vec![]),
//                 Typespec::Any
//             ]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_remote_types() {
//         let code = r#"
//         @type type :: String.t()
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::RemoteType(
//                 "String".to_string(),
//                 "t".to_string(),
//                 vec![]
//             ))
//         ))]);

//         assert_eq!(result, target);

//         let code = r#"
//         @type type :: String.t
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_remote_types_with_params() {
//         let code = r#"
//         @type type :: Enum.to_list(t)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::RemoteType(
//                 "Enum".to_string(),
//                 "to_list".to_string(),
//                 vec![Typespec::LocalType("t".to_string(), vec![])]
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_map() {
//         let code = r#"
//         @type type :: %{a: integer()}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("type".to_string(), vec![])),
//             Box::new(Typespec::Map(vec![(
//                 Typespec::Atom("a".to_string()),
//                 Typespec::LocalType("integer".to_string(), vec![])
//             )]))
//         ))]);

//         assert_eq!(result, target);

//         let code = r#"
//         @type type :: %{:a => integer()}
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_parameters() {
//         let code = r#"
//         @type my_type(t, s) :: {t, s}
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType(
//                 "my_type".to_string(),
//                 vec![
//                     Typespec::LocalType("t".to_string(), vec![]),
//                     Typespec::LocalType("s".to_string(), vec![])
//                 ]
//             )),
//             Box::new(Typespec::Tuple(vec![
//                 Typespec::LocalType("t".to_string(), vec![]),
//                 Typespec::LocalType("s".to_string(), vec![])
//             ]))
//         ))]);
//         assert_eq!(result, target);
//     }

//     /// TODO: Evaluate and add more test cases if necessary
//     #[test]
//     fn parse_spec_definition() {
//         let code = r#"
//         @spec my_func(integer(), String.t()) :: atom()
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![spec!(Typespec::Binding(
//             Box::new(Typespec::LocalType(
//                 "my_func".to_string(),
//                 vec![
//                     Typespec::LocalType("integer".to_string(), vec![]),
//                     Typespec::RemoteType("String".to_string(), "t".to_string(), vec![])
//                 ]
//             )),
//             Box::new(Typespec::LocalType("atom".to_string(), vec![]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_with() {
//         let code = r#"
//         with true <- func?() do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![with_pattern!(bool!(true), call_no_args!(id!("func?")))],
//             else_clauses: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         with true <- func?(), do: :ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_with_multiple_clauses() {
//         let code = r#"
//         with true <- func?(),
//              {:ok, result} <- other_func() do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![
//                 with_pattern!(bool!(true), call_no_args!(id!("func?"))),
//                 with_pattern!(
//                     tuple!(atom!("ok"), id!("result")),
//                     call_no_args!(id!("other_func"))
//                 ),
//             ],
//             else_clauses: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         with true <- func?(),
//              {:ok, result} <- other_func(), do: :ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_with_binding_clauses() {
//         let code = r#"
//         with a = to_string(x),
//              {:ok, result} <- other_func(a) do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![
//                 WithClause::Binding(WithPattern {
//                     pattern: id!("a"),
//                     source: call!(id!("to_string"), id!("x")),
//                 }),
//                 with_pattern!(
//                     tuple!(atom!("ok"), id!("result")),
//                     call!(id!("other_func"), id!("a"))
//                 ),
//             ],
//             else_clauses: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         with a = to_string(x),
//              {:ok, result} <- other_func(a), do: :ok
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_with_else() {
//         let code = r#"
//         with {:ok, result} <- other_func(a) do
//             :ok
//         else
//             {:error, reason} -> reason
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![with_pattern!(
//                 tuple!(atom!("ok"), id!("result")),
//                 call!(id!("other_func"), id!("a"))
//             )],
//             else_clauses: Some(vec![Stab {
//                 left: vec![tuple!(atom!("error"), id!("reason"))],
//                 right: vec![id!("reason")],
//             }]),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_with_guard() {
//         let code = r#"
//         with {:ok, result} when is_integer(result) <- other_func(a) do
//             :ok
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![WithClause::Match(WithPattern {
//                 pattern: Expression::Guard(Guard {
//                     left: Box::new(tuple!(atom!("ok"), id!("result"))),
//                     right: Box::new(call!(id!("is_integer"), id!("result"))),
//                 }),
//                 source: call!(id!("other_func"), id!("a")),
//             })],
//             else_clauses: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_with_else_guards() {
//         let code = r#"
//         with {:ok, result} <- other_func(a) do
//             :ok
//         else
//             {:error, reason} when is_integer(reason) -> reason
//             _else -> :unknown_error
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::With(WithExpression {
//             do_block: Box::new(atom!("ok")),
//             match_clauses: vec![with_pattern!(
//                 tuple!(atom!("ok"), id!("result")),
//                 call!(id!("other_func"), id!("a"))
//             )],
//             else_clauses: Some(vec![
//                 Stab {
//                     left: vec![Expression::Guard(Guard {
//                         left: Box::new(tuple!(atom!("error"), id!("reason"))),
//                         right: Box::new(call!(id!("is_integer"), id!("reason"))),
//                     })],
//                     right: vec![id!("reason")],
//                 },
//                 Stab {
//                     left: vec![id!("_else")],
//                     right: vec![atom!("unknown_error")],
//                 },
//             ]),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_for() {
//         let code = r#"
//         for n <- [1, 2, 3] do
//             n * 2
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::For(For {
//             generators: vec![ForGenerator::Generator(
//                 id!("n"),
//                 list!(int!(1), int!(2), int!(3)),
//             )],
//             block: Box::new(_do!(binary_operation!(
//                 id!("n"),
//                 BinaryOperator::Mult,
//                 int!(2)
//             ))),
//             uniq: None,
//             into: None,
//         })]);

//         assert_eq!(result, target);

//         let code = r#"
//         for n <- [1, 2, 3], do: n * 2
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);

//         let code = r#"
//         for repo <- repos() do
//             repo
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::For(For {
//             generators: vec![ForGenerator::Generator(
//                 id!("repo"),
//                 call_no_args!(id!("repos")),
//             )],
//             block: Box::new(_do!(id!("repo"))),
//             uniq: None,
//             into: None,
//         })]);
//         assert_eq!(result, target);

//         let code = r#"
//         for repo <- repos(), do: repo
//         "#;
//         let result = parse(&code).unwrap();
//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_for_multiple_generators() {
//         let code = r#"
//         for n <- a, a <- [[1, 2], [3, 4]] do
//             n * 2
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::For(For {
//             generators: vec![
//                 ForGenerator::Generator(id!("n"), id!("a")),
//                 ForGenerator::Generator(
//                     id!("a"),
//                     list!(list!(int!(1), int!(2)), list!(int!(3), int!(4))),
//                 ),
//             ],
//             block: Box::new(_do!(binary_operation!(
//                 id!("n"),
//                 BinaryOperator::Mult,
//                 int!(2)
//             ))),
//             uniq: None,
//             into: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_for_with_filters() {
//         let code = r#"
//         for n <- [1, 2, 3], rem(n, 2) == 0 do
//             n * 2
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::For(For {
//             generators: vec![
//                 ForGenerator::Generator(id!("n"), list!(int!(1), int!(2), int!(3))),
//                 ForGenerator::Filter(binary_operation!(
//                     call!(id!("rem"), id!("n"), int!(2)),
//                     BinaryOperator::Equal,
//                     int!(0)
//                 )),
//             ],
//             block: Box::new(_do!(binary_operation!(
//                 id!("n"),
//                 BinaryOperator::Mult,
//                 int!(2)
//             ))),
//             uniq: None,
//             into: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_for_binding() {
//         let code = r#"
//         for n <- [1, 2, 3], x = n + 1 do
//             n * 2
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::For(For {
//             generators: vec![
//                 ForGenerator::Generator(id!("n"), list!(int!(1), int!(2), int!(3))),
//                 ForGenerator::Match(
//                     id!("x"),
//                     binary_operation!(id!("n"), BinaryOperator::Plus, int!(1)),
//                 ),
//             ],
//             block: Box::new(_do!(binary_operation!(
//                 id!("n"),
//                 BinaryOperator::Mult,
//                 int!(2)
//             ))),
//             uniq: None,
//             into: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     // TODO: parse reduce for
//     // TODO: parse :uniq option for
//     // TODO: parse :into option for
//     // TODO: parse for with options

//     #[test]
//     fn parse_char_question_operator() {
//         let code = "?A";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Charlist("A".to_string())]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_char() {
//         let code = "'a'";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Charlist("a".to_string())]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_charlist() {
//         let code = "'test charlist'";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Charlist("test charlist".to_string())]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typep_attribute() {
//         let code = "
//         @typep private :: integer()
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![typep!(Typespec::Binding(
//             Box::new(Typespec::LocalType("private".to_string(), vec![])),
//             Box::new(Typespec::LocalType("integer".to_string(), vec![]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_opaque_attribute() {
//         let code = "
//         @opaque opaque :: atom()
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![opaque!(Typespec::Binding(
//             Box::new(Typespec::LocalType("opaque".to_string(), vec![])),
//             Box::new(Typespec::LocalType("atom".to_string(), vec![]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_callback_attribute() {
//         let code = "
//         @callback extensions() :: [String.t]
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![callback!(Typespec::Binding(
//             Box::new(Typespec::LocalType("extensions".to_string(), vec![])),
//             Box::new(Typespec::List(vec![Typespec::RemoteType(
//                 "String".to_string(),
//                 "t".to_string(),
//                 vec![]
//             )]))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_macrocallback_attribute() {
//         let code = "
//         @macrocallback generate(a) :: list(tuple())
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![macrocallback!(Typespec::Binding(
//             Box::new(Typespec::LocalType(
//                 "generate".to_string(),
//                 vec![Typespec::LocalType("a".to_string(), vec![])]
//             )),
//             Box::new(Typespec::LocalType(
//                 "list".to_string(),
//                 vec![Typespec::LocalType("tuple".to_string(), vec![])]
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_typespec_no_args() {
//         let code = "
//         @type test :: (-> any())
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("test".to_string(), vec![])),
//             Box::new(Typespec::Lambda(
//                 vec![],
//                 Box::new(Typespec::LocalType("any".to_string(), vec![]))
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_typespec_simple() {
//         let code = "
//         @type test :: (... -> integer())
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("test".to_string(), vec![])),
//             Box::new(Typespec::Lambda(
//                 vec![Typespec::Any],
//                 Box::new(Typespec::LocalType("integer".to_string(), vec![]))
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_typespec_multiple_params() {
//         let code = "
//         @type test :: (integer(), string() -> atom())
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("test".to_string(), vec![])),
//             Box::new(Typespec::Lambda(
//                 vec![
//                     Typespec::LocalType("integer".to_string(), vec![]),
//                     Typespec::LocalType("string".to_string(), vec![])
//                 ],
//                 Box::new(Typespec::LocalType("atom".to_string(), vec![]))
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_nonempty_struct() {
//         let code = "
//         @type t() :: %__MODULE__{:name => binary(), age: integer()}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("t".to_string(), vec![])),
//             Box::new(Typespec::Struct(
//                 "__MODULE__".to_string(),
//                 vec![
//                     (
//                         Typespec::Atom("name".to_string()),
//                         Typespec::LocalType("binary".to_string(), vec![])
//                     ),
//                     (
//                         Typespec::Atom("age".to_string()),
//                         Typespec::LocalType("integer".to_string(), vec![])
//                     ),
//                 ]
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_empty_struct() {
//         let code = "
//         @type t() :: %__MODULE__{}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("t".to_string(), vec![])),
//             Box::new(Typespec::Struct("__MODULE__".to_string(), vec![],))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_struct_module_name() {
//         let code = "
//         @type t() :: %Ecto.Changeset{}
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("t".to_string(), vec![])),
//             Box::new(Typespec::Struct("Ecto.Changeset".to_string(), vec![],))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_typespec_remote_erlang_type() {
//         let code = "
//         @type t() :: :uri_string.uri_string()
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![_type!(Typespec::Binding(
//             Box::new(Typespec::LocalType("t".to_string(), vec![])),
//             Box::new(Typespec::RemoteType(
//                 "uri_string".to_string(),
//                 "uri_string".to_string(),
//                 vec![]
//             ))
//         ))]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_var_call() {
//         let code = "
//         func.()
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::AnonymousCall(Call {
//             target: Box::new(id!("func")),
//             remote_callee: None,
//             arguments: vec![],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_lambda_expression_call() {
//         let code = "
//         (fn a -> a + 1 end).(10)
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::AnonymousCall(Call {
//             target: Box::new(Expression::Grouping(Box::new(Expression::Lambda(Lambda {
//                 clauses: vec![Stab {
//                     left: vec![id!("a")],
//                     right: vec![binary_operation!(id!("a"), BinaryOperator::Plus, int!(1))],
//                 }],
//             })))),
//             remote_callee: None,
//             arguments: vec![int!(10)],
//             do_block: None,
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_defguard() {
//         let code = "
//         defguard is_redirect(status) when is_integer(status) and status >= 300 and status < 400
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Defguard(CustomGuard {
//             name: "is_redirect".to_string(),
//             body: Box::new(binary_operation!(
//                 binary_operation!(
//                     call!(id!("is_integer"), id!("status")),
//                     BinaryOperator::StrictAnd,
//                     binary_operation!(id!("status"), BinaryOperator::GreaterThanOrEqual, int!(300))
//                 ),
//                 BinaryOperator::StrictAnd,
//                 binary_operation!(id!("status"), BinaryOperator::LessThan, int!(400))
//             )),
//             parameters: vec![id!("status")],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_defguardp() {
//         let code = "
//         defguardp is_ten(value) when value == 10
//         ";
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Defguardp(CustomGuard {
//             name: "is_ten".to_string(),
//             body: Box::new(binary_operation!(
//                 id!("value"),
//                 BinaryOperator::Equal,
//                 int!(10)
//             )),
//             parameters: vec![id!("value")],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_function_head() {
//         let code = r#"
//         def my_func(a, b \\ nil, c \\ 10)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::FunctionHead(FunctionHead {
//             name: "my_func".to_string(),
//             is_private: false,
//             parameters: vec![
//                 id!("a"),
//                 binary_operation!(id!("b"), BinaryOperator::Default, nil!()),
//                 binary_operation!(id!("c"), BinaryOperator::Default, int!(10)),
//             ],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_stab() {
//         let code = r#"
//         quote do
//             _else -> true
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(_do!(Expression::Stab(Stab {
//                 left: vec![id!("_else")],
//                 right: vec![bool!(true)],
//             }))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_multiple_args_left_stab() {
//         let code = r#"
//         quote do
//             20, 10 -> true
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(_do!(Expression::Stab(Stab {
//                 left: vec![int!(20), int!(10)],
//                 right: vec![bool!(true)],
//             }))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_multiple_expr_block_right_stab() {
//         let code = r#"
//         quote do
//             "string" ->
//                 a = 1
//                 a
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(_do!(Expression::Stab(Stab {
//                 left: vec![string!("string")],
//                 right: vec![
//                     binary_operation!(id!("a"), BinaryOperator::Match, int!(1)),
//                     id!("a")
//                 ],
//             }))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_guarded_stab() {
//         let code = r#"
//         quote do
//             var when is_integer(var) -> var + 1
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Quote(Quote {
//             options: None,
//             body: Box::new(_do!(Expression::Stab(Stab {
//                 left: vec![Expression::Guard(Guard {
//                     left: Box::new(id!("var")),
//                     right: Box::new(call!(id!("is_integer"), id!("var")))
//                 })],
//                 right: vec![binary_operation!(id!("var"), BinaryOperator::Plus, int!(1))]
//             }))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_try_rescue() {
//         let code = r#"
//         try do
//             1 / 0
//         rescue
//             err -> {:error, err}
//         end
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Call(Call {
//             target: Box::new(id!("try")),
//             remote_callee: None,
//             do_block: Some(Do {
//                 body: Box::new(binary_operation!(int!(1), BinaryOperator::Div, int!(0))),
//                 optional_block: Some(OptionalBlock {
//                     block: Box::new(Expression::Stab(Stab {
//                         left: vec![id!("err")],
//                         right: vec![tuple!(atom!("error"), id!("err"))],
//                     })),
//                     block_type: OptionalBlockType::Rescue,
//                 }),
//             }),
//             arguments: vec![],
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_guard() {
//         let code = r#"
//         a when is_integer(a)
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Guard(Guard {
//             left: Box::new(id!("a")),
//             right: Box::new(call!(id!("is_integer"), id!("a"))),
//         })]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_call_keyword_list_args() {
//         let code = r#"
//         defdelegate size(cache_name \\ name()), to: Cache.Adapter
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![
//             call!(
//                 id!("defdelegate"),
//                 call!(
//                     id!("size"),
//                     binary_operation!(
//                         id!("cache_name"),
//                         BinaryOperator::Default,
//                         call_no_args!(id!("name"))
//                     )
//                 )
//             ),
//             list!(tuple!(atom!("to"), atom!("Cache.Adapter"))),
//         ]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_default_expression() {
//         let code = r#"
//         cache_name \\ "my_cache"
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![binary_operation!(
//             id!("cache_name"),
//             BinaryOperator::Default,
//             string!("my_cache")
//         )]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_simple_bitstring() {
//         let code = r#"
//         <<10, 20, 30>>
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Bitstring(vec![
//             int!(10),
//             int!(20),
//             int!(30),
//         ])]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_empty_bitstring() {
//         let code = r#"
//         <<>>
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Bitstring(vec![])]);

//         assert_eq!(result, target);
//     }

//     #[test]
//     fn parse_bitstring_size_specifier() {
//         let code = r#"
//         <<10::8, 20::size(8)>>
//         "#;
//         let result = parse(&code).unwrap();
//         let target = Block(vec![Expression::Bitstring(vec![
//             binary_operation!(int!(10), BinaryOperator::Type, int!(8)),
//             binary_operation!(int!(20), BinaryOperator::Type, call!(id!("size"), int!(8))),
//         ])]);

//         assert_eq!(result, target);
//     }
// }

// TODO: parse typespec as having expression body rather than specific typespec body
// we should not try to be stricter as the elixir compiler, if it parses it's valid for us

// TODO: ... as macro argument (TODO: probably treat it as a identifier if not inside typespec)
//       base_db_query()
//      |> where([..., site], site.domain == ^domain)

// TODO: support calls in types
//   @type role() :: unquote(Enum.reduce(@roles, &{:|, [], [&1, &2]}))

// TODO: failing to parse
// # @spec export_queries(pos_integer,
// #         extname: String.t(),
// #         date_range: Date.Range.t(),
// #         timezone: String.t()
// #       ) ::
// #         %{String.t() => Ecto.Query.t()}

// TODO: support specs with guards for @specs
// https://hexdocs.pm/elixir/typespecs.html#defining-a-specification
// @spec function(arg) :: [arg] when arg: var

// TODO: separate remote from local calls in the Expression enum?
//
// TODO: parse for with more forms and options (do is not necessarily on the end)
// TODO: for site <- sites, do: {site.id, site.domain}, into: %{}
//
// TODO: allow trailing commas in list, tuples, maps, bitstrings
//
// TODO: Math."++add++"(1, 2)
//
// TODO: parse bitstrings
//
// TODO: fix nested kw list bug

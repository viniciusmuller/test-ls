mod indexer;
mod parser;
mod simple_parser;

use std::{
    env,
    ffi::OsStr,
    fs,
    time::{Duration, Instant},
};

use simple_parser::Expression;
use walkdir::WalkDir;

fn main() {
    let args: Vec<String> = env::args().collect();

    let paths = &[&args[1]]; // fs::read_dir(&args[1]).unwrap();

    for path in paths {
        let now = Instant::now();
        let mut results = walkdir(path.as_str());
        let total_elapsed = now.elapsed();

        results.sort_by(|(_, _, e1), (_, _, e2)| e2.cmp(e1));

        println!("Top 10 slowest files to parse:");
        for (path, _, duration) in results.iter().take(10) {
            println!("{:?} - {:.2?}", path, duration);
        }

        let total_modules = results
            .iter()
            .fold(0, |acc, (_, expr, _)| acc + count_expression_modules(expr));
        let total_functions = results.iter().fold(0, |acc, (_, expr, _)| {
            acc + count_expression_functions(expr)
        });

        println!("finished indexing: {} files", results.len());
        println!("Indexed {} modules", total_modules);
        println!("Indexed {} functions", total_functions);
        println!("Total reading + parsing time: {:.2?}", total_elapsed);
    }
}

// TODO: create general structure for folding the ast on different criteria
fn count_expression_modules(ast: &Expression) -> usize {
    match ast {
        Expression::Module(module) => 1 + count_expression_modules(&module.body),
        Expression::Attribute(_, _) => 0,
        Expression::FunctionDef(function) => count_expression_modules(&function.body),
        Expression::String(_) => 0,
        Expression::Scope(scope) => scope
            .body
            .iter()
            .fold(0, |acc, expr| acc + count_expression_modules(&expr)),
        Expression::Identifier(_) => 0,
        Expression::Unparsed(_, children) => children
            .iter()
            .fold(0, |acc, expr| acc + count_expression_modules(&expr)),
        Expression::TreeSitterError(_, _) => 0,
    }
}

fn count_expression_functions(ast: &Expression) -> usize {
    match ast {
        Expression::Module(module) => count_expression_functions(&module.body),
        Expression::Attribute(_, _) => 0,
        Expression::String(_) => 0,
        Expression::FunctionDef(function) => 1 + count_expression_functions(&function.body),
        Expression::Scope(scope) => scope
            .body
            .iter()
            .fold(0, |acc, expr| acc + count_expression_functions(&expr)),
        Expression::Identifier(_) => 0,
        Expression::Unparsed(_, children) => children
            .iter()
            .fold(0, |acc, expr| acc + count_expression_functions(&expr)),
        Expression::TreeSitterError(_, _) => 0,
    }
}

fn walkdir(path: &str) -> Vec<(String, Expression, Duration)> {
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

    file_paths
        .par_iter()
        .map(|path| {
            let contents =
                fs::read_to_string(path).expect("Should have been able to read the file");
            let now = Instant::now();
            let parser = simple_parser::Parser::new(contents, path.to_str().unwrap().to_owned());
            let result = parser.parse();
            let elapsed = now.elapsed();
            (path.to_str().unwrap().to_owned(), result, elapsed)
        })
        .collect::<Vec<_>>()
}

use crate::wasm_backend::wasm_codegen;
use clap::{Parser, Subcommand, arg};
use colored::Colorize;
use frontend::ast::Program;
use frontend::sema::{Analyzer, SemaError};
use frontend::{jam_include, parser};
use logos::Span;
use std::{collections::HashSet, fs, path::PathBuf};

mod frontend;
mod wasm_backend;

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct JamcArgs {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Compile(CompileArgs),
    Import(ImportArgs),
}

#[derive(clap::Parser, Debug, Clone)]
struct ImportArgs {
    #[arg(short, long)]
    inc_dir: PathBuf,

    #[arg(short, long)]
    name: String,
}

#[derive(clap::Parser, Debug, Clone)]
struct CompileArgs {
    #[arg(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
    files: Vec<PathBuf>,

    #[arg(short, long)]
    lib_dir: PathBuf,

    #[arg(short, long)]
    inc_dir: PathBuf,

    #[arg(short, long)]
    target_triple: String,

    #[arg(value_parser, num_args = 0.., value_delimiter = ' ', last=true)]
    linker_args: Vec<String>,

    #[arg(short, long)]
    output_name: String,

    #[arg(long, action)]
    debug: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
struct SourceFile {
    id: usize,
    module_name: String,
    module_path: PathBuf,
    program: Program,
    source: String,
}

impl SourceFile {
    // TODO: Maybe cache this so I don't need to iter over source a bunch
    // Also TODO: Use this before panicing in codegen
    #[allow(dead_code)]
    pub(crate) fn get_line_for_span(&self, span: &Span) -> (usize, usize) {
        let mut line_start = 0;
        let mut _line_number = 1;

        for (i, line) in self.source.lines().enumerate() {
            let line_end = line_start + line.len();

            if span.start >= line_start && span.start <= line_end {
                let col = span.start - line_start + 1;
                return (i, col);
            }

            line_start = line_end + get_newline_len(&self.source, line_end); // +1 for '\n' +2 for \r\n
            _line_number += 1;
        }
        eprintln!("Failed to file source line for {:?}", span);
        return (0, 0);
    }
}

fn main() {
    let start = std::time::Instant::now();

    let args = JamcArgs::parse();

    let compile_args = match args.cmd {
        Command::Compile(compile_args) => compile_args,
        _ => panic!("unsupported command"),
    };
    let input_files: Vec<PathBuf> = compile_args.files.iter().map(PathBuf::from).collect();

    let mut source_files = Vec::new();
    //let mut program_id = 0_usize;

    println!(
        "{: >10} {}",
        "Compiling".bright_green(),
        &compile_args.output_name
    );

    let mut has_error = false;

    let intrinsics = include_str!("intrinsics.jam");
    // This should never fail
    let program = parser::Parser::parse_program(intrinsics, 0).unwrap();

    source_files.push(SourceFile {
        id: 0,
        module_name: "intrinsics".to_string(),
        module_path: PathBuf::new(),
        program: program,
        source: intrinsics.to_string(),
    });
    for file in &input_files {
        let module_name = file
            .file_name()
            .expect("failed to get file name")
            .to_str()
            .unwrap();
        let src =
            fs::read_to_string(file).expect(&format!("failed to read input file <{:?}>", &file));

        let program_id = source_files.len();

        match parser::Parser::parse_program(&src, program_id) {
            Ok(p) => {
                source_files.push(SourceFile {
                    module_name: module_name.to_owned(),
                    module_path: file.to_path_buf(),
                    id: program_id,
                    source: src,
                    program: p,
                });
            }
            Err(e) => {
                report_parse_error(&e, &src, &module_name);
                has_error = true;
            }
        }
    }

    let all_includes: Vec<frontend::ast::IncludeStmt> = source_files
        .iter()
        .flat_map(|src| &src.program.items)
        .filter_map(|item| {
            if let frontend::ast::Item::IncludeStmt(stmt) = item {
                Some(stmt.clone())
            } else {
                None
            }
        })
        .collect();

    let all_includes: HashSet<frontend::ast::IncludeStmt> =
        HashSet::from_iter(all_includes.iter().map(|i| i.clone()));

    let mut seen_modules: HashSet<PathBuf> =
        compile_args.files.iter().map(|f| f.to_owned()).collect();

    let mut next_id = source_files.last().expect("no files?").id + 1;
    if !jam_include::include_includes2(
        &mut source_files,
        &all_includes,
        &compile_args.inc_dir,
        &mut seen_modules,
        &mut next_id,
    ) {
        has_error = true;
    }

    if has_error {
        panic!("could not compile module due to the previous errors");
    }

    let analyzer: Analyzer = Analyzer::new(&source_files);
    let (_sym_table, ty_table, static_strings, errors, addr_of_uses_by_function) =
        analyzer.run(&source_files);

    if errors.is_empty() {
        let Ok(_) = wasm_codegen::emit_wasm(
            &source_files,
            &ty_table,
            &addr_of_uses_by_function,
            static_strings,
        ) else {
            print!("{} module failed to validate!", "error:".red());
            panic!()
        };
        println!(
            "{: >10} [{}]{}{} in {:?}",
            "Finished".bright_green(),
            compile_args.target_triple,
            if compile_args.debug {
                " (debug) ".purple()
            } else {
                "".purple()
            },
            &compile_args.output_name,
            start.elapsed()
        );
    } else {
        let error = errors.first().unwrap();
        report_sema_error(error, &source_files[error.source_id]);
        panic!("could not compile module due to the previous errors");
    }
}

fn report_parse_error(e: &parser::ParseError, src: &str, module_name: &str) {
    let source = &src;
    let span = &e.span;
    let message = &e.message;
    let file_name = &module_name;

    report_error(source, span, message, file_name);
}

fn report_sema_error(sema_error: &SemaError, source_file: &SourceFile) {
    let source = &source_file.source;
    let span = &sema_error.span;
    let message = &sema_error.message;
    let file_name = &source_file.module_name;

    report_error(source, span, message, file_name);
}

pub fn report_error(source: &str, span: &Span, message: &String, file_name: &str) {
    let mut line_start = 0;
    let mut _line_number = 1;

    for (i, line) in source.lines().enumerate() {
        let line_end = line_start + line.len();

        if span.start >= line_start && span.start <= line_end {
            let col = span.start - line_start + 1;

            println!();

            println!("{}: {}", "error".red(), message.white());
            println!(" --> {}:{}:{}", file_name, i + 1, col + 1);
            println!("  {}", "|".blue());
            println!("{: >3} {} {}", i + 1, "|".blue(), line); // line number
            println!("  {} {:>col$}{}", "|".blue(), "", "^".red(), col = col);
            return;
        }

        line_start = line_end + get_newline_len(source, line_end); // +1 for '\n' +2 for \r\n
        _line_number += 1;
    }

    // fallback if out of bounds
    println!("error: {} (at byte {})", message, span.start);
}

fn get_newline_len(source: &str, start_at_exclusive: usize) -> usize {
    let bytes = source.as_bytes();

    if start_at_exclusive + 1 < bytes.len()
        && bytes[start_at_exclusive] == b'\r'
        && bytes[start_at_exclusive + 1] == b'\n'
    {
        return 2;
    } else if start_at_exclusive < bytes.len()
        && (bytes[start_at_exclusive] == b'\n' || bytes[start_at_exclusive] == b'\r')
    {
        return 1;
    }

    0
}

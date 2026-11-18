//! Command-line runner and interactive `>>>` prompt for femtopython.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use femtopython::{Config, InputStatus, Interpreter, Value, input_status};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

/// A small, strictly-typed, embeddable subset of Python.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Script to run. Starts an interactive prompt if omitted.
    file: Option<PathBuf>,

    /// Run CODE instead of a script file.
    #[arg(short = 'c', value_name = "CODE", conflicts_with = "file")]
    command: Option<String>,

    /// Enter the interactive prompt after running FILE or CODE.
    #[arg(short = 'i')]
    interactive: bool,

    /// Allow scripts to read files with open().
    #[arg(long)]
    allow_fs: bool,

    /// Maximum evaluation nesting depth (expressions plus calls).
    #[arg(long, value_name = "N", default_value_t = Config::default().max_eval_depth)]
    max_eval_depth: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let mut interp = Interpreter::with_config(Config {
        allow_fs: args.allow_fs,
        max_eval_depth: args.max_eval_depth,
        ..Config::default()
    });

    let script = match (&args.command, &args.file) {
        (Some(code), _) => Some(code.clone()),
        (None, Some(path)) => match std::fs::read_to_string(path) {
            Ok(src) => Some(src),
            Err(e) => {
                eprintln!("femtopython: cannot read {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        (None, None) => None,
    };

    if let Some(src) = &script {
        if let Err(e) = interp.run(src) {
            eprintln!("{}", e.render(src));
            if !args.interactive {
                return ExitCode::FAILURE;
            }
        }
        if !args.interactive {
            return ExitCode::SUCCESS;
        }
    }

    match repl(&mut interp) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("femtopython: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A `def` needs a terminating blank line, as in CPython's REPL.
fn starts_block(src: &str) -> bool {
    src.trim_start()
        .strip_prefix("def")
        .is_some_and(|rest| rest.starts_with(char::is_whitespace))
}

fn repl(interp: &mut Interpreter) -> Result<(), ReadlineError> {
    let mut rl = DefaultEditor::new()?;
    println!("femtopython {} (Ctrl-D to exit)", env!("CARGO_PKG_VERSION"));
    'outer: loop {
        let mut buf = String::new();
        let mut prompt = ">>> ";
        loop {
            let line = match rl.readline(prompt) {
                Ok(line) => line,
                Err(ReadlineError::Interrupted) => {
                    println!("KeyboardInterrupt");
                    continue 'outer;
                }
                Err(ReadlineError::Eof) => return Ok(()),
                Err(e) => return Err(e),
            };
            buf.push_str(&line);
            buf.push('\n');
            if buf.trim().is_empty() {
                continue 'outer;
            }
            match input_status(&buf) {
                InputStatus::Incomplete => {}
                InputStatus::Complete if starts_block(&buf) && !line.trim().is_empty() => {}
                _ => break,
            }
            prompt = "... ";
        }
        let _ = rl.add_history_entry(buf.trim_end());
        match interp.run(&buf) {
            Ok(Value::None) => {}
            Ok(v) => println!("{}", v.repr()),
            Err(e) => eprintln!("{}", e.render(&buf)),
        }
    }
}

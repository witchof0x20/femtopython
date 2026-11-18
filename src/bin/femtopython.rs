// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

//! Command-line runner and interactive `>>>` prompt for femtopython.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use clap::Parser;
use femtopython::{
    Allowlist, Config, Error, ErrorKind, InputStatus, Interpreter, Killer, Opener, Value,
    input_status,
};
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

    /// Disable open() entirely.
    #[arg(long, conflicts_with = "allow_path")]
    no_fs: bool,

    /// Only let open() read this file, or files beneath this directory.
    /// Repeatable. Without it, open() reads any file, like Python.
    #[arg(long, value_name = "PATH")]
    allow_path: Vec<PathBuf>,

    /// Maximum evaluation nesting depth (expressions plus calls).
    #[arg(long, value_name = "N", default_value_t = Config::default().max_eval_depth)]
    max_eval_depth: usize,

    /// Maximum number of steps per run (unlimited by default; Ctrl-C
    /// interrupts a running script, a second Ctrl-C exits).
    #[arg(long, value_name = "N")]
    max_steps: Option<u64>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let opener = match opener(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("femtopython: {e}");
            return ExitCode::from(2);
        }
    };
    let interp = Interpreter::with_config(Config {
        opener,
        max_eval_depth: args.max_eval_depth,
        max_steps: args.max_steps,
        ..Config::default()
    });
    let mut runner = match Runner::new(interp) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("femtopython: cannot install Ctrl-C handler: {e}");
            return ExitCode::FAILURE;
        }
    };

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
        if let Err(e) = runner.run(src) {
            report(&e, src);
            if !args.interactive {
                return match e.kind {
                    // 128 + SIGINT, the shell convention for Ctrl-C.
                    ErrorKind::Killed => ExitCode::from(130),
                    _ => ExitCode::FAILURE,
                };
            }
        }
        if !args.interactive {
            return ExitCode::SUCCESS;
        }
    }

    match repl(&mut runner) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("femtopython: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Runs each input on its own thread (`Interpreter::spawn`) so Ctrl-C can
/// kill it at the next step and the prompt carries on with the globals the
/// script left behind.
struct Runner {
    interp: Option<Interpreter>,
    running: Arc<Mutex<Running>>,
}

/// The script currently running, shared with the Ctrl-C handler.
#[derive(Default)]
struct Running {
    /// Kills the running script; `None` while waiting for input.
    killer: Option<Killer>,
    /// Ctrl-C was already pressed during this run.
    kill_sent: bool,
}

impl Runner {
    fn new(interp: Interpreter) -> Result<Runner, ctrlc::Error> {
        let running: Arc<Mutex<Running>> = Arc::default();
        let handler_running = running.clone();
        ctrlc::set_handler(move || {
            let mut r = handler_running.lock().unwrap_or_else(|e| e.into_inner());
            let Running { killer, kill_sent } = &mut *r;
            match killer {
                // A second Ctrl-C while the first hasn't taken effect (e.g.
                // the script is stuck inside a native function): give up.
                Some(_) if *kill_sent => std::process::exit(130),
                Some(k) => {
                    *kill_sent = true;
                    k.kill();
                }
                None => {}
            }
        })?;
        Ok(Runner {
            interp: Some(interp),
            running,
        })
    }

    fn set_running(&self, killer: Option<Killer>) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = Running {
            killer,
            kill_sent: false,
        };
    }

    fn run(&mut self, src: &str) -> Result<Value, Error> {
        let handle = self.interp.take().expect("not running").spawn(src);
        self.set_running(Some(handle.killer()));
        let (interp, result) = handle.join();
        self.set_running(None);
        self.interp = Some(interp);
        result
    }
}

fn report(e: &Error, src: &str) {
    match e.kind {
        ErrorKind::Killed => eprintln!("KeyboardInterrupt"),
        _ => eprintln!("{}", e.render(src)),
    }
}

fn opener(args: &Args) -> std::io::Result<Opener> {
    if args.no_fs {
        return Ok(Opener::deny());
    }
    if args.allow_path.is_empty() {
        return Ok(Opener::filesystem());
    }
    let mut list = Allowlist::new();
    for p in &args.allow_path {
        list = if p.is_dir() {
            list.dir(p)
        } else {
            list.file(p)
        }
        .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", p.display())))?;
    }
    Ok(Opener::allowlist(list))
}

/// Compound statements need a terminating blank line, as in CPython's REPL.
fn starts_block(src: &str) -> bool {
    let src = src.trim_start();
    ["def", "if", "while", "for"].iter().any(|kw| {
        src.strip_prefix(kw)
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_whitespace() || c == '('))
    })
}

fn repl(runner: &mut Runner) -> Result<(), ReadlineError> {
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
        match runner.run(&buf) {
            Ok(Value::None) => {}
            Ok(v) => println!("{}", v.repr()),
            Err(e) => report(&e, &buf),
        }
    }
}

// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end tests of the `femtopython` binary: Ctrl-C (a real SIGINT)
//! interrupts a running script.
#![cfg(all(unix, feature = "repl"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

fn femtopython(args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_femtopython"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Read stdout until a line equal to `marker`, returning everything read.
fn read_until(out: &mut BufReader<ChildStdout>, marker: &str) -> String {
    let mut seen = String::new();
    loop {
        let mut line = String::new();
        assert!(
            out.read_line(&mut line).unwrap() > 0,
            "EOF before {marker:?}; got {seen:?}"
        );
        seen.push_str(&line);
        if line.trim_end().ends_with(marker) {
            return seen;
        }
    }
}

fn sigint(child: &Child) {
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0
    );
}

/// Wait for exit, failing the test (and killing the child) if it hangs.
fn wait(mut child: Child) -> (std::process::ExitStatus, String, String) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            panic!("femtopython did not exit after Ctrl-C");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut s) = child.stdout.take() {
        s.read_to_string(&mut out).unwrap();
    }
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    (status, out, err)
}

const LOOP: &str = "n = 0
while True:
    if n == 0:
        print('ready')
    n += 1
";

#[test]
fn ctrl_c_interrupts_a_script() {
    let mut child = femtopython(&["-c", LOOP]);
    let mut out = BufReader::new(child.stdout.take().unwrap());
    read_until(&mut out, "ready");
    sigint(&child);
    let (status, _, err) = wait(child);
    assert_eq!(status.code(), Some(130), "stderr: {err}");
    assert!(err.contains("KeyboardInterrupt"), "stderr: {err}");
}

#[test]
fn ctrl_c_in_the_repl_interrupts_and_keeps_going() {
    let mut child = femtopython(&[]);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    // A blank line ends the `while` block, as in Python's REPL.
    writeln!(stdin, "{LOOP}").unwrap();
    stdin.flush().unwrap();
    read_until(&mut out, "ready");
    sigint(&child);
    // The REPL is still alive, and globals survived the interrupt.
    writeln!(stdin, "print('after', n > 0)").unwrap();
    stdin.flush().unwrap();
    let rest = read_until(&mut out, "after True");
    assert!(rest.contains("after True"), "{rest}");
    // Interrupt it again, to check repeated Ctrl-C works.
    writeln!(stdin, "{}", LOOP.replace("n = 0", "n = 0\nm = 1")).unwrap();
    stdin.flush().unwrap();
    read_until(&mut out, "ready");
    sigint(&child);
    writeln!(stdin, "print('m is', m)").unwrap();
    drop(stdin); // EOF: exit
    read_until(&mut out, "m is 1");
    let (status, _, err) = wait(child);
    assert!(status.success(), "stderr: {err}");
    assert_eq!(err.matches("KeyboardInterrupt").count(), 2, "stderr: {err}");
}

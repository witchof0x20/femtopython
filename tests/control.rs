// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

use std::sync::Arc;
use std::time::{Duration, Instant};

use femtopython::{Config, ErrorKind, Interpreter, Progress, StepControl, Value};

const FOREVER: &str = "i = 0\nwhile True:\n    i += 1";

/// No step limit, so only a kill can stop `FOREVER`.
fn unlimited() -> Interpreter {
    Interpreter::with_config(Config {
        max_steps: None,
        ..Config::default()
    })
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn finishes_normally() {
    let (interp, r) = Interpreter::new().spawn("x = 20\nx * 2 + 2").join();
    assert_eq!(r.unwrap(), Value::I64(42));
    assert_eq!(interp.get("x"), Some(Value::I64(20)));
}

#[test]
fn kill_a_running_script() {
    let h = unlimited().spawn(FOREVER);
    wait_until("the script to start", || h.steps() > 1000);
    h.kill();
    let (interp, r) = h.join();
    assert!(matches!(r.unwrap_err().kind, ErrorKind::Killed));
    // Globals are as the script left them.
    assert!(matches!(interp.get("i"), Some(Value::I64(n)) if n > 0));
}

#[test]
fn kill_a_paused_script() {
    let h = unlimited().spawn_paused(FOREVER);
    assert_eq!(h.pause(), Progress::Paused);
    assert_eq!(h.steps(), 0);
    h.kill();
    let (_, r) = h.join();
    assert!(matches!(r.unwrap_err().kind, ErrorKind::Killed));
}

#[test]
fn kill_from_another_thread() {
    let h = unlimited().spawn(FOREVER);
    let killer = h.killer();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        killer.kill();
    });
    let (_, r) = h.join();
    assert!(matches!(r.unwrap_err().kind, ErrorKind::Killed));
}

#[test]
fn dropping_the_handle_kills_the_script() {
    // A native function's captured Arc is released when the interpreter
    // (and so the script thread) is gone.
    let alive = Arc::new(());
    let mut interp = unlimited();
    let token = alive.clone();
    interp.register_fn("tick", move || {
        let _ = &token;
    });
    let h = interp.spawn("while True:\n    tick()");
    wait_until("the script to start", || h.steps() > 100);
    drop(h);
    wait_until("the thread to exit", || Arc::strong_count(&alive) == 1);
}

#[test]
fn single_stepping() {
    let h = unlimited().spawn_paused(FOREVER);
    assert_eq!(h.step(10), Progress::Paused);
    assert_eq!(h.steps(), 10);
    assert_eq!(h.step(5), Progress::Paused);
    assert_eq!(h.steps(), 15);
    // Paused means paused.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(h.steps(), 15);
    // Resume, then pause again.
    h.resume();
    wait_until("progress", || h.steps() > 1000);
    assert_eq!(h.pause(), Progress::Paused);
    let n = h.steps();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(h.steps(), n);
    h.kill();
    assert!(h.join().1.is_err());
}

#[test]
fn stepping_past_the_end_finishes() {
    let h = Interpreter::new().spawn_paused("a = 1\nb = a + 1\nb");
    assert_eq!(h.step(1), Progress::Paused);
    assert_eq!(h.step(1000), Progress::Finished);
    assert!(h.is_finished());
    assert_eq!(h.step(1), Progress::Finished);
    assert_eq!(h.join().1.unwrap(), Value::I64(2));
}

#[test]
fn spawn_with_calls_functions() {
    let mut interp = Interpreter::new();
    interp.run("def sq(n: int) -> int: return n * n").unwrap();
    let sq = interp.get("sq").unwrap();
    let h = interp.spawn_with(false, move |i| i.call(&sq, vec![Value::I64(9)]));
    assert_eq!(h.join().1.unwrap(), Value::I64(81));
}

#[test]
fn native_panics_become_errors() {
    let mut interp = Interpreter::new();
    interp.register_fn("boom", || -> i64 { panic!("kaboom") });
    let (interp, r) = interp.spawn("x = 1\nboom()").join();
    match r.unwrap_err().kind {
        ErrorKind::Panic(m) => assert_eq!(m, "kaboom"),
        other => panic!("{other:?}"),
    }
    // The interpreter is still usable, globals intact.
    assert_eq!(interp.get("x"), Some(Value::I64(1)));
}

#[test]
fn custom_step_hook() {
    let mut interp = unlimited();
    let mut budget = 50;
    interp.set_step_hook(move || {
        budget -= 1;
        if budget == 0 {
            StepControl::Stop
        } else {
            StepControl::Continue
        }
    });
    let e = interp.run(FOREVER).unwrap_err();
    assert!(matches!(e.kind, ErrorKind::Killed));
    interp.clear_step_hook();
    assert_eq!(interp.run("1").unwrap(), Value::I64(1));
}

#[test]
fn step_limit_still_applies_on_a_thread() {
    let h = Interpreter::with_config(Config {
        max_steps: Some(100),
        ..Config::default()
    })
    .spawn(FOREVER);
    assert!(matches!(
        h.join().1.unwrap_err().kind,
        ErrorKind::Limit(femtopython::LimitExceeded::Steps(100))
    ));
}

// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

//! Running a script on its own thread, with pause, single-stepping and kill.
//!
//! ```
//! use femtopython::{ErrorKind, Interpreter, Progress};
//!
//! let handle = Interpreter::new().spawn_paused("i = 0\nwhile True:\n    i += 1");
//! assert_eq!(handle.step(100), Progress::Paused); // runs exactly 100 steps
//! handle.kill();
//! let (interp, result) = handle.join();
//! assert!(matches!(result.unwrap_err().kind, ErrorKind::Killed));
//! assert!(interp.get("i").is_some()); // globals survive the kill
//! ```

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::error::{Error, ErrorKind, Result};
use crate::interp::{Interpreter, StepControl};
use crate::value::Value;

struct State {
    /// Steps the script may still take before pausing; `None` = unlimited.
    budget: Option<u64>,
    killed: bool,
    finished: bool,
    /// Steps taken so far.
    steps: u64,
    /// Bumped by every `step`/`pause` request; recorded by the script
    /// thread when it pauses, so a request only completes on a pause that
    /// happened after it.
    epoch: u64,
    paused_at: Option<u64>,
}

struct Control {
    state: Mutex<State>,
    cv: Condvar,
}

impl Control {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, g: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        self.cv.wait(g).unwrap_or_else(|e| e.into_inner())
    }

    /// The step hook, run on the script thread.
    fn on_step(&self) -> StepControl {
        let mut s = self.lock();
        loop {
            if s.killed {
                return StepControl::Stop;
            }
            match s.budget {
                None => break,
                Some(0) => {
                    s.paused_at = Some(s.epoch);
                    self.cv.notify_all();
                    s = self.wait(s);
                }
                Some(n) => {
                    s.budget = Some(n - 1);
                    break;
                }
            }
        }
        s.paused_at = None;
        s.steps += 1;
        StepControl::Continue
    }

    fn kill(&self) {
        self.lock().killed = true;
        self.cv.notify_all();
    }
}

/// Where a script is after [`ScriptHandle::step`] / [`ScriptHandle::pause`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// Stopped between steps, waiting to be stepped, resumed or killed.
    Paused,
    /// Done (successfully, with an error, or killed); call `join`.
    Finished,
}

/// Kills a spawned script from anywhere (another thread, a signal handler,
/// an async task). Cheap to clone.
#[derive(Clone)]
pub struct Killer(Arc<Control>);

impl Killer {
    /// Stop the script at its next step, whether it is running or paused.
    /// It finishes with [`ErrorKind::Killed`].
    pub fn kill(&self) {
        self.0.kill();
    }
}

/// A script running on its own thread. Dropping the handle without calling
/// [`join`](Self::join) kills the script.
///
/// Kills take effect at the next step (statement, loop iteration or
/// iterator item), so a script can always be stopped promptly, unless it is
/// inside a host function that blocks: that can't be interrupted until it
/// returns.
pub struct ScriptHandle {
    control: Arc<Control>,
    thread: Option<JoinHandle<(Interpreter, Result<Value>)>>,
}

impl ScriptHandle {
    /// Run exactly `n` more steps, then pause. Blocks until the script has
    /// paused or finished.
    pub fn step(&self, n: u64) -> Progress {
        let mut s = self.control.lock();
        if s.finished {
            return Progress::Finished;
        }
        s.budget = Some(n);
        s.epoch += 1;
        let epoch = s.epoch;
        self.control.cv.notify_all();
        loop {
            if s.finished {
                return Progress::Finished;
            }
            if s.paused_at.is_some_and(|e| e >= epoch) {
                return Progress::Paused;
            }
            s = self.control.wait(s);
        }
    }

    /// Pause at the next step. Blocks until the script has paused or
    /// finished.
    pub fn pause(&self) -> Progress {
        self.step(0)
    }

    /// Let the script run freely. Doesn't block.
    pub fn resume(&self) {
        self.control.lock().budget = None;
        self.control.cv.notify_all();
    }

    /// Stop the script at its next step. Doesn't block; use
    /// [`join`](Self::join) to wait for it.
    pub fn kill(&self) {
        self.control.kill();
    }

    pub fn killer(&self) -> Killer {
        Killer(self.control.clone())
    }

    pub fn is_finished(&self) -> bool {
        self.control.lock().finished
    }

    /// Steps taken so far.
    pub fn steps(&self) -> u64 {
        self.control.lock().steps
    }

    /// Wait for the script to finish and get the interpreter back, with its
    /// globals as the script left them (also after an error or a kill).
    pub fn join(mut self) -> (Interpreter, Result<Value>) {
        let thread = self.thread.take().expect("joined once");
        match thread.join() {
            Ok(r) => r,
            // The script closure itself is run under `catch_unwind`, so this
            // only happens if the bookkeeping around it panicked.
            Err(p) => std::panic::resume_unwind(p),
        }
    }
}

impl Drop for ScriptHandle {
    fn drop(&mut self) {
        if self.thread.is_some() {
            // Not joined: stop the script so the thread can't run forever.
            // The thread is detached and exits at the script's next step.
            self.control.kill();
        }
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

fn spawn(
    mut interp: Interpreter,
    paused: bool,
    f: impl FnOnce(&mut Interpreter) -> Result<Value> + Send + 'static,
) -> ScriptHandle {
    let control = Arc::new(Control {
        state: Mutex::new(State {
            budget: paused.then_some(0),
            killed: false,
            finished: false,
            steps: 0,
            epoch: 0,
            paused_at: None,
        }),
        cv: Condvar::new(),
    });
    let ctl = control.clone();
    let thread = std::thread::Builder::new()
        .name("femtopython".into())
        .spawn(move || {
            let hook_ctl = ctl.clone();
            interp.set_step_hook(move || hook_ctl.on_step());
            let result = catch_unwind(AssertUnwindSafe(|| f(&mut interp)))
                .unwrap_or_else(|p| Err(Error::new(ErrorKind::Panic(panic_message(&*p)))));
            interp.clear_step_hook();
            ctl.lock().finished = true;
            ctl.cv.notify_all();
            (interp, result)
        })
        .expect("failed to spawn interpreter thread");
    ScriptHandle {
        control,
        thread: Some(thread),
    }
}

impl Interpreter {
    /// Run `src` on a new thread, starting immediately. See
    /// [`ScriptHandle`] for pausing, stepping and killing it.
    pub fn spawn(self, src: impl Into<String>) -> ScriptHandle {
        let src = src.into();
        spawn(self, false, move |i| i.run(&src))
    }

    /// Like [`spawn`](Self::spawn), but paused before the first step.
    pub fn spawn_paused(self, src: impl Into<String>) -> ScriptHandle {
        let src = src.into();
        spawn(self, true, move |i| i.run(&src))
    }

    /// Run any interpreter operation (e.g. [`call`](Self::call)) on a new
    /// thread, under the same controls.
    pub fn spawn_with(
        self,
        paused: bool,
        f: impl FnOnce(&mut Interpreter) -> Result<Value> + Send + 'static,
    ) -> ScriptHandle {
        spawn(self, paused, f)
    }
}

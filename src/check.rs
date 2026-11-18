//! Static analysis.
//!
//! * [`check_program`] validates a whole program before any of it runs:
//!   `return` only inside a `def`, `break`/`continue` only inside a loop,
//!   unique parameter names.
//! * [`free_vars_of_def`] / [`free_vars_of_lambda`] compute, at parse time,
//!   which outer variables a function body reads and writes. When the `def`
//!   executes, exactly those are captured, and lock permissions are checked
//!   against them.

use std::collections::HashSet;

use crate::ast::{
    Expr, ExprKind, FreeVars, Param, READ_WRITE_METHODS, Stmt, StmtKind, is_mutating_method,
    target_path,
};
use crate::error::{Result, err};

pub fn check_program(stmts: &[Stmt]) -> Result<()> {
    let ctx = Ctx {
        in_function: false,
        in_loop: false,
    };
    stmts.iter().try_for_each(|s| check_stmt(s, ctx))
}

#[derive(Clone, Copy)]
struct Ctx {
    in_function: bool,
    in_loop: bool,
}

fn check_block(stmts: &[Stmt], ctx: Ctx) -> Result<()> {
    stmts.iter().try_for_each(|s| check_stmt(s, ctx))
}

fn check_stmt(s: &Stmt, ctx: Ctx) -> Result<()> {
    let at = |e: crate::Error| e.at(s.span);
    match &s.kind {
        StmtKind::Expr(e) => check_expr(e),
        StmtKind::Declare { value, .. } => check_expr(value),
        StmtKind::Assign { target, value } | StmtKind::AugAssign { target, value, .. } => {
            check_expr(target)?;
            check_expr(value)
        }
        StmtKind::Return(e) => {
            if !ctx.in_function {
                return Err(at(err!(Syntax, "'return' outside function")));
            }
            e.as_ref().map_or(Ok(()), check_expr)
        }
        StmtKind::Break | StmtKind::Continue if !ctx.in_loop => {
            let kw = if matches!(s.kind, StmtKind::Break) {
                "break"
            } else {
                "continue"
            };
            Err(at(err!(Syntax, "'{kw}' outside loop")))
        }
        StmtKind::Pass | StmtKind::Break | StmtKind::Continue => Ok(()),
        StmtKind::Def(def) => {
            let mut seen = HashSet::new();
            for p in &def.params {
                if !seen.insert(&p.name) {
                    return Err(err!(Syntax, "duplicate parameter '{}'", p.name).at(p.span));
                }
            }
            check_block(
                &def.body,
                Ctx {
                    in_function: true,
                    in_loop: false,
                },
            )
        }
        StmtKind::If { branches, orelse } => {
            for (cond, body) in branches {
                check_expr(cond)?;
                check_block(body, ctx)?;
            }
            orelse.as_ref().map_or(Ok(()), |b| check_block(b, ctx))
        }
        StmtKind::While { cond, body } => {
            check_expr(cond)?;
            check_block(
                body,
                Ctx {
                    in_loop: true,
                    ..ctx
                },
            )
        }
        StmtKind::For {
            targets,
            iter,
            body,
        } => {
            let mut seen = HashSet::new();
            for t in targets {
                if !seen.insert(t) {
                    return Err(at(err!(Syntax, "duplicate loop variable '{t}'")));
                }
            }
            check_expr(iter)?;
            check_block(
                body,
                Ctx {
                    in_loop: true,
                    ..ctx
                },
            )
        }
    }
}

fn check_expr(e: &Expr) -> Result<()> {
    if let ExprKind::Lambda(l) = &e.kind {
        let mut seen = HashSet::new();
        for p in &l.params {
            if !seen.insert(p) {
                return Err(err!(Syntax, "duplicate parameter '{p}'").at(e.span));
            }
        }
    }
    let mut result = Ok(());
    e.kind.for_each_child(|c| {
        if result.is_ok() {
            result = check_expr(c);
        }
    });
    result
}

// ------------------------------------------------------- free variables

/// Walks a function body in program order with a stack of block scopes,
/// mirroring the runtime's scoping rules, and records every use of a name
/// that is not (yet) declared locally.
struct Collector {
    scopes: Vec<HashSet<String>>,
    free: FreeVars,
}

impl Collector {
    fn new(params: impl IntoIterator<Item = String>) -> Self {
        Collector {
            scopes: vec![params.into_iter().collect()],
            free: FreeVars::default(),
        }
    }

    fn declared(&self, n: &str) -> bool {
        self.scopes.iter().any(|s| s.contains(n))
    }

    fn read(&mut self, n: &str) {
        if !self.declared(n) {
            self.free.reads.insert(n.to_owned());
        }
    }

    fn write(&mut self, n: &str) {
        if !self.declared(n) {
            self.free.writes.insert(n.to_owned());
        }
    }

    fn declare(&mut self, n: &str) {
        self.scopes.last_mut().unwrap().insert(n.to_owned());
    }

    fn nested(&mut self, free: &FreeVars) {
        for n in &free.reads {
            self.read(n);
        }
        for n in &free.writes {
            self.write(n);
        }
    }

    fn block(&mut self, stmts: &[Stmt], declare_first: &[String]) {
        self.scopes.push(declare_first.iter().cloned().collect());
        for s in stmts {
            self.stmt(s);
        }
        self.scopes.pop();
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Expr(e) => self.expr(e),
            StmtKind::Declare {
                name,
                ty: Some(_),
                value,
            } => {
                self.expr(value);
                self.declare(name);
            }
            StmtKind::Declare {
                name,
                ty: None,
                value,
            } => {
                self.expr(value);
                // Either writes a captured outer variable or declares a new
                // local; capturing it if it exists covers both cases.
                if !self.declared(name) {
                    self.free.writes.insert(name.clone());
                    self.declare(name);
                }
            }
            StmtKind::Assign { target, value } => {
                self.expr(value);
                self.target(target, false);
            }
            StmtKind::AugAssign { target, value, .. } => {
                self.expr(value);
                self.target(target, true);
            }
            StmtKind::Return(e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Def(def) => {
                self.declare(&def.name);
                self.nested(&def.free);
            }
            StmtKind::If { branches, orelse } => {
                for (cond, body) in branches {
                    self.expr(cond);
                    self.block(body, &[]);
                }
                if let Some(body) = orelse {
                    self.block(body, &[]);
                }
            }
            StmtKind::While { cond, body } => {
                self.expr(cond);
                self.block(body, &[]);
            }
            StmtKind::For {
                targets,
                iter,
                body,
            } => {
                self.expr(iter);
                self.block(body, targets);
            }
        }
    }

    /// An assignment target: the root is written (and read, for `op=`); the
    /// index expressions are ordinary reads.
    fn target(&mut self, target: &Expr, also_read: bool) {
        let (root, idxs) = target_path(target).expect("parser only builds valid targets");
        for i in idxs {
            self.expr(i);
        }
        if also_read {
            self.read(root);
        }
        self.write(root);
    }

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Name(n) => self.read(n),
            ExprKind::Lambda(l) => self.nested(&l.free),
            ExprKind::Call(callee, args) => {
                match &callee.kind {
                    ExprKind::Attr(obj, m) if is_mutating_method(m) => match target_path(obj) {
                        Some(_) => self.target(obj, READ_WRITE_METHODS.contains(&m.as_str())),
                        None => self.expr(obj),
                    },
                    _ => self.expr(callee),
                }
                for a in args {
                    self.expr(a);
                }
            }
            kind => kind.for_each_child(|c| self.expr(c)),
        }
    }
}

pub fn free_vars_of_def(params: &[Param], body: &[Stmt]) -> FreeVars {
    let mut c = Collector::new(params.iter().map(|p| p.name.clone()));
    for s in body {
        c.stmt(s);
    }
    c.free
}

pub fn free_vars_of_lambda(params: &[String], body: &Expr) -> FreeVars {
    let mut c = Collector::new(params.iter().cloned());
    c.expr(body);
    c.free
}

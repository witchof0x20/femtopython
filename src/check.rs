//! Static validation run on the whole program before any of it executes.
//!
//! Because femtopython has no control flow, function bodies are straight-line
//! code and these properties are decidable exactly:
//!
//! * `return` only appears inside a `def`.
//! * Inside a function, a name may be declared at most once (redeclaration is
//!   a top-level-only feature), parameters count as declarations, and
//!   parameter names are unique.
//! * Inside a function, `+=`-style and subscript assignments only target the
//!   function's own locals: captured variables are read-only.

use std::collections::HashSet;

use crate::ast::{Expr, ExprKind, Stmt, StmtKind, target_path};
use crate::error::{Result, err};

pub fn check_program(stmts: &[Stmt]) -> Result<()> {
    for s in stmts {
        check_stmt(s, None)?;
    }
    Ok(())
}

fn check_stmt(s: &Stmt, mut locals: Option<&mut HashSet<String>>) -> Result<()> {
    let at = |e: crate::Error| e.at(s.span);
    match &s.kind {
        StmtKind::Expr(e) => check_expr(e),
        StmtKind::Declare { name, value, .. } => {
            check_expr(value)?;
            if let Some(locals) = locals
                && !locals.insert(name.clone())
            {
                return Err(at(err!(
                    Syntax,
                    "'{name}' is already declared in this function; redeclaration is only \
                     allowed at the top level (use an augmented assignment like += instead)"
                )));
            }
            Ok(())
        }
        StmtKind::Assign { target, value } | StmtKind::AugAssign { target, value, .. } => {
            let (root, path) = target_path(target).expect("parser only builds valid targets");
            for idx in path {
                check_expr(idx)?;
            }
            check_expr(value)?;
            if let Some(locals) = locals
                && !locals.contains(root)
            {
                return Err(err!(
                    Syntax,
                    "cannot assign to '{root}': it is not a local variable of this function \
                     (captured and global variables are read-only inside functions)"
                )
                .at(target.span));
            }
            Ok(())
        }
        StmtKind::Return(e) => {
            if locals.is_none() {
                return Err(at(err!(Syntax, "'return' outside function")));
            }
            e.as_ref().map_or(Ok(()), check_expr)
        }
        StmtKind::Pass => Ok(()),
        StmtKind::Def(def) => {
            if let Some(locals) = locals.as_mut()
                && !locals.insert(def.name.clone())
            {
                return Err(at(err!(
                    Syntax,
                    "'{}' is already declared in this function",
                    def.name
                )));
            }
            let mut inner = HashSet::new();
            for p in &def.params {
                if !inner.insert(p.name.clone()) {
                    return Err(err!(Syntax, "duplicate parameter '{}'", p.name).at(p.span));
                }
            }
            for s in &def.body {
                check_stmt(s, Some(&mut inner))?;
            }
            Ok(())
        }
    }
}

fn check_expr(e: &Expr) -> Result<()> {
    match &e.kind {
        ExprKind::Lambda(l) => {
            let mut seen = HashSet::new();
            for p in &l.params {
                if !seen.insert(p) {
                    return Err(err!(Syntax, "duplicate parameter '{p}'").at(e.span));
                }
            }
            check_expr(&l.body)
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Bytes(_)
        | ExprKind::Bool(_)
        | ExprKind::None
        | ExprKind::Name(_) => Ok(()),
        ExprKind::List(es) | ExprKind::Tuple(es) | ExprKind::Set(es) => {
            es.iter().try_for_each(check_expr)
        }
        ExprKind::Dict(kvs) => kvs.iter().try_for_each(|(k, v)| {
            check_expr(k)?;
            check_expr(v)
        }),
        ExprKind::Unary(_, x) | ExprKind::Attr(x, _) => check_expr(x),
        ExprKind::Binary(_, a, b)
        | ExprKind::Compare(_, a, b)
        | ExprKind::Logic(_, a, b)
        | ExprKind::Index(a, b) => {
            check_expr(a)?;
            check_expr(b)
        }
        ExprKind::Call(f, args) => {
            check_expr(f)?;
            args.iter().try_for_each(check_expr)
        }
    }
}

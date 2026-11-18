//! Abstract syntax tree.
//!
//! Expressions are depth-limited at construction ([`Expr::new`]), so the
//! parser can never hand the evaluator (or `Drop`) a tree deep enough to
//! overflow the native stack.

use std::rc::Rc;

use crate::error::{Error, Span, err};
use crate::types::Type;

/// Maximum nesting depth of a single expression. Matches CPython's parser
/// limit on nested parentheses in spirit.
pub const MAX_EXPR_DEPTH: u32 = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Pos,
    Invert,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    depth: u32,
}

#[derive(Debug)]
pub enum ExprKind {
    /// Unsigned magnitude; the sign comes from an enclosing `Neg`, and the
    /// concrete integer type from context (see `Interpreter::eval_hinted`).
    Int(u64),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<[u8]>),
    Bool(bool),
    None,
    Name(String),
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    Set(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    Logic(LogicOp, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Attr(Box<Expr>, String),
    Lambda(Rc<LambdaDef>),
}

#[derive(Debug)]
pub struct LambdaDef {
    pub params: Vec<String>,
    pub body: Expr,
}

impl Expr {
    /// Build an expression node, rejecting it if it would exceed
    /// [`MAX_EXPR_DEPTH`].
    pub fn new(kind: ExprKind, span: Span) -> Result<Expr, Error> {
        let mut child_depth = 0;
        kind.for_each_child(|c| child_depth = child_depth.max(c.depth));
        let depth = child_depth + 1;
        if depth > MAX_EXPR_DEPTH {
            return Err(err!(
                Limit,
                "expression nesting exceeds the maximum depth of {MAX_EXPR_DEPTH}"
            )
            .at(span));
        }
        Ok(Expr { kind, span, depth })
    }

    /// Literal numbers (optionally signed) adopt their type from context and
    /// have no side effects, so they may be evaluated after their siblings.
    pub fn is_numeric_literal(&self) -> bool {
        match &self.kind {
            ExprKind::Int(_) | ExprKind::Float(_) => true,
            ExprKind::Unary(UnOp::Neg | UnOp::Pos, e) => e.is_numeric_literal(),
            _ => false,
        }
    }
}

impl ExprKind {
    fn for_each_child(&self, mut f: impl FnMut(&Expr)) {
        match self {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bytes(_)
            | ExprKind::Bool(_)
            | ExprKind::None
            | ExprKind::Name(_) => {}
            ExprKind::List(es) | ExprKind::Tuple(es) | ExprKind::Set(es) => es.iter().for_each(f),
            ExprKind::Dict(kvs) => kvs.iter().for_each(|(k, v)| {
                f(k);
                f(v)
            }),
            ExprKind::Unary(_, e) | ExprKind::Attr(e, _) => f(e),
            ExprKind::Binary(_, a, b)
            | ExprKind::Compare(_, a, b)
            | ExprKind::Logic(_, a, b)
            | ExprKind::Index(a, b) => {
                f(a);
                f(b)
            }
            ExprKind::Call(callee, args) => {
                f(callee);
                args.iter().for_each(f)
            }
            ExprKind::Lambda(l) => f(&l.body),
        }
    }
}

#[derive(Debug)]
pub struct Param {
    pub name: String,
    pub ty: Option<Type>,
    pub span: Span,
}

#[derive(Debug)]
pub struct FuncDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Vec<Stmt>,
}

#[derive(Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug)]
pub enum StmtKind {
    Expr(Expr),
    /// `name = value` or `name: ty = value`. A (re)declaration, never a
    /// mutation.
    Declare {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    /// `target[index]... = value`. `target` is always an `Index` chain rooted
    /// at a `Name` (enforced by the parser).
    Assign {
        target: Expr,
        value: Expr,
    },
    /// `target op= value`. `target` is a `Name` or an `Index` chain.
    AugAssign {
        target: Expr,
        op: BinOp,
        value: Expr,
    },
    Return(Option<Expr>),
    Pass,
    Def(Rc<FuncDef>),
}

/// For an assignment target, the root variable name and the chain of index
/// expressions applied to it (outermost last). `None` if the expression is
/// not assignable.
pub fn target_path(e: &Expr) -> Option<(&str, Vec<&Expr>)> {
    match &e.kind {
        ExprKind::Name(n) => Some((n, Vec::new())),
        ExprKind::Index(base, idx) => {
            let (root, mut path) = target_path(base)?;
            path.push(idx);
            Some((root, path))
        }
        _ => None,
    }
}

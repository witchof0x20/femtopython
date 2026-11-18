//! Tree-walking evaluator and the embeddable [`Interpreter`] object.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read as _, Write};
use std::rc::Rc;

use crate::ast::{
    BinOp, Expr, ExprKind, FuncDef, LambdaDef, LogicOp, Stmt, StmtKind, UnOp, target_path,
};
use crate::error::{ErrorKind, Result, err};
use crate::native::{IntoNativeFn, NativeFn};
use crate::ops;
use crate::types::Type;
use crate::value::{FileHandle, Key, Value};

const STACK_RED_ZONE: usize = 128 * 1024;
const STACK_SEGMENT: usize = 2 * 1024 * 1024;

/// Host-controlled limits and capabilities.
#[derive(Debug, Clone)]
pub struct Config {
    /// Whether scripts may call `open`. Off by default: a script can only
    /// touch the filesystem if the embedder opts in.
    pub allow_fs: bool,
    /// Maximum evaluation nesting (expression depth plus call depth). Bounds
    /// native stack usage, so runaway recursion is an error rather than a
    /// crash of the host process.
    pub max_eval_depth: usize,
    /// Maximum length in bytes of any string/bytes value produced by
    /// concatenation or file reads.
    pub max_str_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            allow_fs: false,
            max_eval_depth: 400,
            max_str_len: 16 * 1024 * 1024,
        }
    }
}

/// A user-defined function or lambda.
pub struct Function {
    pub name: String,
    params: Vec<(String, Option<Type>)>,
    ret: Option<Type>,
    body: Body,
    /// Snapshot of the enclosing function's variables at definition time
    /// (like a Rust `move` closure). Globals are not captured; they are
    /// looked up live, which is what lets top-level functions recurse and
    /// call each other.
    captured: Rc<HashMap<String, Value>>,
}

enum Body {
    Block(Rc<FuncDef>),
    Lambda(Rc<LambdaDef>),
}

impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<function {}>", self.name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Builtin {
    Print,
    Map,
    Open,
    /// A type used as a conversion function, e.g. `u8(x)` or `Path("a")`.
    Convert(Type),
}

impl Builtin {
    fn lookup(name: &str) -> Option<Builtin> {
        Some(match name {
            "print" => Builtin::Print,
            "map" => Builtin::Map,
            "open" => Builtin::Open,
            "bool" | "i8" | "i16" | "i32" | "i64" | "int" | "u8" | "u16" | "u32" | "u64"
            | "f32" | "f64" | "float" | "str" | "bytes" | "Path" => {
                Builtin::Convert(Type::from_name(name, vec![]).expect("known type name"))
            }
            _ => return None,
        })
    }

    pub fn name(&self) -> String {
        match self {
            Builtin::Print => "print".into(),
            Builtin::Map => "map".into(),
            Builtin::Open => "open".into(),
            Builtin::Convert(t) => t.to_string(),
        }
    }
}

/// Local state of one function call.
struct Frame {
    locals: HashMap<String, Value>,
    captured: Rc<HashMap<String, Value>>,
    current: Rc<Function>,
}

enum Flow {
    Next(Option<Value>),
    Return(Value),
}

/// Whether a chunk of source is ready to run. Useful for building REPLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputStatus {
    Complete,
    /// More lines are needed (open bracket, empty `def` body, ...).
    Incomplete,
    /// The input is wrong no matter what follows; running it reports why.
    Invalid,
}

/// Classify `src` without running it.
pub fn input_status(src: &str) -> InputStatus {
    match crate::parser::parse_program(src) {
        Ok(_) => InputStatus::Complete,
        Err(e) if e.kind == ErrorKind::IncompleteInput => InputStatus::Incomplete,
        Err(_) => InputStatus::Invalid,
    }
}

/// An embeddable femtopython interpreter. Global variables persist across
/// calls to [`run`](Interpreter::run) and [`eval`](Interpreter::eval).
///
/// ```
/// use femtopython::{Interpreter, Value};
///
/// let mut interp = Interpreter::new();
/// interp.set("rate", 0.2f64);
/// interp.run("def tax(x: float) -> float: return x * rate").unwrap();
/// assert_eq!(interp.eval("tax(100.0)").unwrap(), Value::F64(20.0));
/// ```
pub struct Interpreter {
    config: Config,
    globals: HashMap<String, Value>,
    out: Box<dyn Write>,
    depth: usize,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Self {
        Interpreter {
            config,
            globals: HashMap::new(),
            out: Box::new(std::io::stdout()),
            depth: 0,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Redirect the output of `print`.
    pub fn set_output(&mut self, out: impl Write + 'static) {
        self.out = Box::new(out);
    }

    /// Run a program (any number of statements). Returns the value of the
    /// final statement if it is an expression, otherwise `None`. If an error
    /// occurs, the effects of statements before it are kept.
    pub fn run(&mut self, src: &str) -> Result<Value> {
        let program = crate::parser::parse_program(src)?;
        crate::check::check_program(&program)?;
        self.depth = 0;
        let mut last = Value::None;
        for stmt in &program {
            last = match self.exec_stmt(stmt, None)? {
                Flow::Next(v) => v.unwrap_or(Value::None),
                Flow::Return(_) => unreachable!("checker rejects top-level return"),
            };
        }
        Ok(last)
    }

    /// Evaluate a single expression.
    pub fn eval(&mut self, src: &str) -> Result<Value> {
        let expr = crate::parser::parse_expr(src)?;
        let stmt = [Stmt {
            span: expr.span,
            kind: StmtKind::Expr(expr),
        }];
        crate::check::check_program(&stmt)?;
        let StmtKind::Expr(expr) = &stmt[0].kind else {
            unreachable!()
        };
        self.depth = 0;
        self.eval_expr(expr, None, None)
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Declare (or redeclare) a global variable.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.globals.insert(name.into(), value.into());
    }

    pub fn globals(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.globals.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Expose a Rust function or closure to scripts as a global. Parameter
    /// and return types are derived from its signature (see
    /// [`IntoNativeFn`] for the supported types):
    ///
    /// ```
    /// use femtopython::{Interpreter, Value};
    ///
    /// let mut interp = Interpreter::new();
    /// interp.register_fn("hypot", |a: f64, b: f64| a.hypot(b));
    /// assert_eq!(interp.eval("hypot(3, 4)").unwrap(), Value::F64(5.0));
    /// assert!(interp.eval("hypot('3', 4)").is_err());
    /// ```
    pub fn register_fn<M>(&mut self, name: impl Into<String>, func: impl IntoNativeFn<M>) {
        let name = name.into();
        let f = func.into_native_fn(name.clone());
        self.globals.insert(name, Value::Native(Rc::new(f)));
    }

    /// Expose an unchecked function that receives the raw argument values,
    /// for variadic or dynamically-typed host functions.
    pub fn register_raw_fn(
        &mut self,
        name: impl Into<String>,
        func: impl Fn(&[Value]) -> Result<Value> + 'static,
    ) {
        let name = name.into();
        let f = NativeFn::raw(name.clone(), func);
        self.globals.insert(name, Value::Native(Rc::new(f)));
    }

    /// Call a script function (e.g. one fetched with [`get`](Self::get))
    /// from the host.
    pub fn call(&mut self, f: &Value, args: Vec<Value>) -> Result<Value> {
        self.depth = 0;
        self.call_value(f, args)
    }

    // ------------------------------------------------------------ statements

    fn exec_stmt(&mut self, stmt: &Stmt, frame: Option<&mut Frame>) -> Result<Flow> {
        let flow = match &stmt.kind {
            StmtKind::Expr(e) => Flow::Next(Some(self.eval_expr(e, frame.as_deref(), None)?)),
            StmtKind::Declare { name, ty, value } => {
                let mut v = self.eval_expr(value, frame.as_deref(), ty.as_ref())?;
                if let Some(ty) = ty {
                    if !ty.accepts(&v.type_of()) {
                        return Err(err!(
                            Type,
                            "'{name}' is declared as '{ty}' but the value has type '{}'",
                            v.type_of()
                        )
                        .at(value.span));
                    }
                    v.refine(ty);
                }
                self.bind(frame, name.clone(), v);
                Flow::Next(None)
            }
            StmtKind::Assign { target, value } => {
                self.assign(frame, target, None, value)?;
                Flow::Next(None)
            }
            StmtKind::AugAssign { target, op, value } => {
                self.assign(frame, target, Some(*op), value)?;
                Flow::Next(None)
            }
            StmtKind::Return(e) => {
                let ret_hint = frame.as_ref().and_then(|f| f.current.ret.clone());
                let v = match e {
                    Some(e) => self.eval_expr(e, frame.as_deref(), ret_hint.as_ref())?,
                    None => Value::None,
                };
                Flow::Return(v)
            }
            StmtKind::Pass => Flow::Next(None),
            StmtKind::Def(def) => {
                let f = Function {
                    name: def.name.clone(),
                    params: def
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.ty.clone()))
                        .collect(),
                    ret: def.ret.clone(),
                    body: Body::Block(def.clone()),
                    captured: capture(frame.as_deref()),
                };
                self.bind(frame, def.name.clone(), Value::Function(Rc::new(f)));
                Flow::Next(None)
            }
        };
        Ok(flow)
    }

    fn bind(&mut self, frame: Option<&mut Frame>, name: String, v: Value) {
        match frame {
            Some(f) => f.locals.insert(name, v),
            None => self.globals.insert(name, v),
        };
    }

    /// `target = value` (subscript targets only) and `target op= value`.
    fn assign(
        &mut self,
        frame: Option<&mut Frame>,
        target: &Expr,
        op: Option<BinOp>,
        value: &Expr,
    ) -> Result<()> {
        let (root, path) = target_path(target).expect("parser only builds valid targets");
        let mut idxs = Vec::with_capacity(path.len());
        for e in &path {
            idxs.push(self.eval_expr(e, frame.as_deref(), None)?);
        }

        let current = match frame.as_deref() {
            Some(f) => f.locals.get(root),
            None => self.globals.get(root),
        }
        .ok_or_else(|| err!(Name, "name '{root}' is not defined").at(target.span))?;
        // The type of the slot being written, used to type literals.
        let mut slot = current.type_of();
        for _ in &idxs {
            slot = match slot {
                Type::List(t) => *t,
                Type::Dict(_, v) => *v,
                _ => Type::Unknown,
            };
        }
        let hint = (slot != Type::Unknown).then_some(slot);
        let rhs = self.eval_expr(value, frame.as_deref(), hint.as_ref())?;

        let max_len = self.config.max_str_len;
        let root_val = match frame {
            Some(f) => f.locals.get_mut(root),
            None => self.globals.get_mut(root),
        }
        .expect("looked up above; evaluation cannot remove variables");
        let span = target.span;
        if idxs.is_empty() {
            let op = op.expect("plain name assignment is a declaration");
            *root_val = ops::binary(op, root_val, &rhs, max_len).map_err(|e| e.at(span))?;
            Ok(())
        } else {
            store(root_val, &idxs, op, rhs, max_len).map_err(|e| e.at(span))
        }
    }

    // ----------------------------------------------------------- expressions

    /// Evaluate `e`. `hint` is the type the context expects; it only affects
    /// numeric literals, which adopt it (as Rust's integer literals do).
    fn eval_expr(&mut self, e: &Expr, frame: Option<&Frame>, hint: Option<&Type>) -> Result<Value> {
        self.depth += 1;
        let r = if self.depth > self.config.max_eval_depth {
            Err(err!(
                Limit,
                "maximum evaluation depth ({}) exceeded; is there unbounded recursion?",
                self.config.max_eval_depth
            ))
        } else {
            // Run on a fresh heap-allocated stack segment when the native
            // stack runs low, so `max_eval_depth` is the only limit and
            // deep-but-allowed evaluation can't overflow the host's thread
            // stack, whatever its size or our frame sizes.
            stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || {
                self.eval_inner(e, frame, hint)
            })
        };
        self.depth -= 1;
        r.map_err(|err| err.at(e.span))
    }

    fn eval_inner(
        &mut self,
        e: &Expr,
        frame: Option<&Frame>,
        hint: Option<&Type>,
    ) -> Result<Value> {
        Ok(match &e.kind {
            ExprKind::Int(n) => int_literal(false, *n, hint)?,
            ExprKind::Float(x) => float_literal(*x, hint),
            ExprKind::Unary(UnOp::Neg, inner) if matches!(inner.kind, ExprKind::Int(_)) => {
                let ExprKind::Int(n) = inner.kind else {
                    unreachable!()
                };
                int_literal(true, n, hint)?
            }
            ExprKind::Str(s) => Value::Str(s.clone()),
            ExprKind::Bytes(b) => Value::Bytes(b.clone()),
            ExprKind::Bool(b) => Value::Bool(*b),
            ExprKind::None => Value::None,
            ExprKind::Name(n) => self.lookup(n, frame)?,
            ExprKind::List(items) => {
                let elem_hint = match hint {
                    Some(Type::List(t)) => Some(&**t),
                    _ => None,
                };
                let refs: Vec<&Expr> = items.iter().collect();
                let vals = self.eval_group(&refs, frame, elem_hint)?;
                let start = if vals.is_empty() {
                    elem_hint.cloned()
                } else {
                    None
                };
                Value::new_list(start.unwrap_or(Type::Unknown), vals)?
            }
            ExprKind::Set(items) => {
                let elem_hint = match hint {
                    Some(Type::Set(t)) => Some(&**t),
                    _ => None,
                };
                let refs: Vec<&Expr> = items.iter().collect();
                let vals = self.eval_group(&refs, frame, elem_hint)?;
                Value::new_set(Type::Unknown, vals)?
            }
            ExprKind::Dict(entries) => {
                let (kh, vh) = match hint {
                    Some(Type::Dict(k, v)) => (Some(&**k), Some(&**v)),
                    _ => (None, None),
                };
                let keys: Vec<&Expr> = entries.iter().map(|(k, _)| k).collect();
                let vals: Vec<&Expr> = entries.iter().map(|(_, v)| v).collect();
                let keys = self.eval_group(&keys, frame, kh)?;
                let vals = self.eval_group(&vals, frame, vh)?;
                let (k0, v0) = if keys.is_empty() {
                    (kh.cloned(), vh.cloned())
                } else {
                    (None, None)
                };
                Value::new_dict(
                    k0.unwrap_or(Type::Unknown),
                    v0.unwrap_or(Type::Unknown),
                    keys.into_iter().zip(vals).collect(),
                )?
            }
            ExprKind::Tuple(items) => {
                let hints: Option<&Vec<Type>> = match hint {
                    Some(Type::Tuple(ts)) if ts.len() == items.len() => Some(ts),
                    _ => None,
                };
                let mut vals = Vec::with_capacity(items.len());
                for (i, item) in items.iter().enumerate() {
                    vals.push(self.eval_expr(item, frame, hints.map(|h| &h[i]))?);
                }
                Value::new_tuple(vals)?
            }
            ExprKind::Unary(op, inner) => {
                let h = if *op == UnOp::Not { None } else { hint };
                let v = self.eval_expr(inner, frame, h)?;
                ops::unary(*op, &v)?
            }
            ExprKind::Binary(op @ (BinOp::Shl | BinOp::Shr), a, b) => {
                let a = self.eval_expr(a, frame, hint)?;
                let b = self.eval_expr(b, frame, None)?;
                ops::binary(*op, &a, &b, self.config.max_str_len)?
            }
            ExprKind::Binary(op, a, b) => {
                let [a, b] = self.eval_pair(a, b, frame, hint)?;
                ops::binary(*op, &a, &b, self.config.max_str_len)?
            }
            ExprKind::Compare(op, a, b) => {
                let [a, b] = self.eval_pair(a, b, frame, None)?;
                Value::Bool(ops::compare(*op, &a, &b)?)
            }
            ExprKind::Logic(op, a, b) => {
                let a_val = self.eval_bool(a, frame, "and/or")?;
                let short = match op {
                    LogicOp::And => !a_val,
                    LogicOp::Or => a_val,
                };
                if short {
                    Value::Bool(a_val)
                } else {
                    Value::Bool(self.eval_bool(b, frame, "and/or")?)
                }
            }
            ExprKind::Call(callee, args) => {
                if let ExprKind::Attr(obj, method) = &callee.kind {
                    let obj = self.eval_expr(obj, frame, None)?;
                    let mut vals = Vec::with_capacity(args.len());
                    for a in args {
                        vals.push(self.eval_expr(a, frame, None)?);
                    }
                    return self.call_method(&obj, method, vals);
                }
                let f = self.eval_expr(callee, frame, None)?;
                let hints = param_hints(&f);
                let mut vals = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    let h = hints.get(i).and_then(Option::as_ref);
                    vals.push(self.eval_expr(a, frame, h)?);
                }
                self.call_value(&f, vals)?
            }
            ExprKind::Index(base, idx) => {
                let base = self.eval_expr(base, frame, None)?;
                let idx_hint = match base.type_of() {
                    Type::Dict(k, _) => Some(*k),
                    _ => None,
                };
                let idx = self.eval_expr(idx, frame, idx_hint.as_ref())?;
                index(&base, &idx)?
            }
            ExprKind::Attr(obj, name) => {
                let obj = self.eval_expr(obj, frame, None)?;
                return Err(err!(
                    Type,
                    "'{}' attribute '{name}' must be called as a method",
                    obj.type_of()
                ));
            }
            ExprKind::Lambda(def) => Value::Function(Rc::new(Function {
                name: "<lambda>".into(),
                params: def.params.iter().map(|p| (p.clone(), None)).collect(),
                ret: None,
                body: Body::Lambda(def.clone()),
                captured: capture(frame),
            })),
        })
    }

    fn eval_bool(&mut self, e: &Expr, frame: Option<&Frame>, what: &str) -> Result<bool> {
        match self.eval_expr(e, frame, None)? {
            Value::Bool(b) => Ok(b),
            other => Err(err!(
                Type,
                "operands of {what} must be bool, got '{}' (there is no implicit truthiness)",
                other.type_of()
            )
            .at(e.span)),
        }
    }

    fn eval_pair(
        &mut self,
        a: &Expr,
        b: &Expr,
        frame: Option<&Frame>,
        hint: Option<&Type>,
    ) -> Result<[Value; 2]> {
        let v = self.eval_group(&[a, b], frame, hint)?;
        Ok(v.try_into().expect("two in, two out"))
    }

    /// Evaluate sibling expressions that must share a type (operands of a
    /// binary operator, elements of a list, ...). Numeric literals are
    /// evaluated last, typed after the first non-literal sibling, else the
    /// context's `hint`, else `float` if any sibling is a float literal.
    /// Literals have no side effects, so the reordering is unobservable.
    fn eval_group(
        &mut self,
        exprs: &[&Expr],
        frame: Option<&Frame>,
        hint: Option<&Type>,
    ) -> Result<Vec<Value>> {
        let mut out: Vec<Option<Value>> = vec![None; exprs.len()];
        let mut found: Option<Type> = None;
        for (i, e) in exprs.iter().enumerate() {
            if !e.is_numeric_literal() {
                let v = self.eval_expr(e, frame, hint)?;
                found.get_or_insert_with(|| v.type_of());
                out[i] = Some(v);
            }
        }
        let lit_ty = found.or_else(|| hint.cloned()).or_else(|| {
            exprs
                .iter()
                .any(|e| e.is_numeric_literal() && is_float_literal(e))
                .then_some(Type::F64)
        });
        for (i, e) in exprs.iter().enumerate() {
            if out[i].is_none() {
                out[i] = Some(self.eval_expr(e, frame, lit_ty.as_ref())?);
            }
        }
        Ok(out.into_iter().map(Option::unwrap).collect())
    }

    fn lookup(&self, name: &str, frame: Option<&Frame>) -> Result<Value> {
        if let Some(f) = frame {
            if let Some(v) = f.locals.get(name) {
                return Ok(v.clone());
            }
            if f.current.name == name {
                return Ok(Value::Function(f.current.clone()));
            }
            if let Some(v) = f.captured.get(name) {
                return Ok(v.clone());
            }
        }
        if let Some(v) = self.globals.get(name) {
            return Ok(v.clone());
        }
        Builtin::lookup(name)
            .map(Value::Builtin)
            .ok_or_else(|| err!(Name, "name '{name}' is not defined"))
    }

    // ----------------------------------------------------------------- calls

    fn call_value(&mut self, f: &Value, args: Vec<Value>) -> Result<Value> {
        match f {
            Value::Function(func) => self.call_function(func, args),
            Value::Builtin(b) => self.call_builtin(b, args),
            Value::Native(n) => n.call(args),
            other => Err(err!(Type, "'{}' is not callable", other.type_of())),
        }
    }

    fn call_function(&mut self, func: &Rc<Function>, args: Vec<Value>) -> Result<Value> {
        if args.len() != func.params.len() {
            return Err(err!(
                Type,
                "{}() takes {} argument(s) but {} were given",
                func.name,
                func.params.len(),
                args.len()
            ));
        }
        let mut locals = HashMap::with_capacity(args.len());
        for ((name, ty), mut arg) in func.params.iter().zip(args) {
            if let Some(ty) = ty {
                if !ty.accepts(&arg.type_of()) {
                    return Err(err!(
                        Type,
                        "{}() argument '{name}' must be '{ty}', not '{}'",
                        func.name,
                        arg.type_of()
                    ));
                }
                arg.refine(ty);
            }
            locals.insert(name.clone(), arg);
        }
        let mut frame = Frame {
            locals,
            captured: func.captured.clone(),
            current: func.clone(),
        };
        let mut result = match &func.body {
            Body::Lambda(l) => self.eval_expr(&l.body, Some(&frame), None)?,
            Body::Block(def) => {
                let mut result = Value::None;
                for stmt in &def.body {
                    if let Flow::Return(v) = self.exec_stmt(stmt, Some(&mut frame))? {
                        result = v;
                        break;
                    }
                }
                result
            }
        };
        if let Some(ret) = &func.ret {
            if !ret.accepts(&result.type_of()) {
                return Err(err!(
                    Type,
                    "{}() must return '{ret}', not '{}'",
                    func.name,
                    result.type_of()
                ));
            }
            result.refine(ret);
        }
        Ok(result)
    }

    fn call_builtin(&mut self, b: &Builtin, args: Vec<Value>) -> Result<Value> {
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(err!(
                    Type,
                    "{}() takes {n} argument(s) but {} were given",
                    b.name(),
                    args.len()
                ))
            }
        };
        match b {
            Builtin::Print => {
                let mut line = String::new();
                for (i, a) in args.iter().enumerate() {
                    if a.is_callable() {
                        return Err(err!(Type, "print() cannot print a function"));
                    }
                    if i > 0 {
                        line.push(' ');
                    }
                    line.push_str(&a.display());
                }
                line.push('\n');
                self.out
                    .write_all(line.as_bytes())
                    .and_then(|_| self.out.flush())
                    .map_err(|e| err!(Io, "print failed: {e}"))?;
                Ok(Value::None)
            }
            Builtin::Map => {
                arity(2)?;
                let f = &args[0];
                if !f.is_callable() {
                    return Err(err!(
                        Type,
                        "map() first argument must be a function, not '{}'",
                        f.type_of()
                    ));
                }
                let items = iterate(&args[1])?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.call_value(f, vec![item])?);
                }
                let elem = match f {
                    Value::Function(func) if out.is_empty() => {
                        func.ret.clone().unwrap_or(Type::Unknown)
                    }
                    Value::Native(n) if out.is_empty() => n.ret().clone(),
                    _ => Type::Unknown,
                };
                Value::new_list(elem, out)
            }
            Builtin::Open => {
                arity(1)?;
                if !self.config.allow_fs {
                    return Err(err!(Permission, "filesystem access is disabled"));
                }
                let path = match &args[0] {
                    Value::Str(s) => std::path::PathBuf::from(&**s),
                    Value::Path(p) => (**p).clone(),
                    other => {
                        return Err(err!(
                            Type,
                            "open() expects str or Path, not '{}'",
                            other.type_of()
                        ));
                    }
                };
                let file = std::fs::File::open(&path)
                    .map_err(|e| err!(Io, "cannot open {}: {e}", path.display()))?;
                Ok(Value::File(Rc::new(std::cell::RefCell::new(FileHandle {
                    path,
                    file,
                }))))
            }
            Builtin::Convert(t) => {
                arity(1)?;
                ops::convert(t, &args[0])
            }
        }
    }

    fn call_method(&mut self, obj: &Value, method: &str, args: Vec<Value>) -> Result<Value> {
        let no_args = || {
            if args.is_empty() {
                Ok(())
            } else {
                Err(err!(Type, "{method}() takes no arguments"))
            }
        };
        match (obj, method) {
            (Value::Dict(d), "keys") => {
                no_args()?;
                Value::new_list(d.key.clone(), d.map.keys().map(Key::to_value).collect())
            }
            (Value::Dict(d), "values") => {
                no_args()?;
                Value::new_list(d.val.clone(), d.map.values().cloned().collect())
            }
            (Value::Dict(d), "items") => {
                no_args()?;
                let elem = Type::Tuple(vec![d.key.clone(), d.val.clone()]);
                let items = d
                    .map
                    .iter()
                    .map(|(k, v)| Value::new_tuple(vec![k.to_value(), v.clone()]))
                    .collect::<Result<_>>()?;
                Value::new_list(elem, items)
            }
            (Value::File(f), "read") => {
                no_args()?;
                let limit = self.config.max_str_len;
                let mut buf = Vec::new();
                let mut fh = f.borrow_mut();
                (&mut fh.file)
                    .take(limit as u64 + 1)
                    .read_to_end(&mut buf)
                    .map_err(|e| err!(Io, "read failed: {e}"))?;
                if buf.len() > limit {
                    return Err(err!(Limit, "file is larger than {limit} bytes"));
                }
                let s = String::from_utf8(buf)
                    .map_err(|e| err!(Value, "file is not valid UTF-8: {e}"))?;
                Ok(Value::Str(s.into()))
            }
            _ => Err(err!(Type, "'{}' has no method '{method}'", obj.type_of())),
        }
    }
}

/// Snapshot the variables visible from `frame` for a new closure.
fn capture(frame: Option<&Frame>) -> Rc<HashMap<String, Value>> {
    let Some(f) = frame else {
        return Rc::new(HashMap::new());
    };
    let mut m = (*f.captured).clone();
    if f.current.name != "<lambda>" {
        m.insert(f.current.name.clone(), Value::Function(f.current.clone()));
    }
    m.extend(f.locals.iter().map(|(k, v)| (k.clone(), v.clone())));
    Rc::new(m)
}

fn param_hints(f: &Value) -> Vec<Option<Type>> {
    match f {
        Value::Function(func) => func.params.iter().map(|(_, t)| t.clone()).collect(),
        Value::Builtin(Builtin::Convert(t)) => vec![Some(t.clone())],
        Value::Native(n) => n
            .params()
            .unwrap_or_default()
            .iter()
            .map(|t| (*t != Type::Unknown).then(|| t.clone()))
            .collect(),
        _ => vec![],
    }
}

fn is_float_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Float(_) => true,
        ExprKind::Unary(_, inner) => is_float_literal(inner),
        _ => false,
    }
}

fn int_literal(neg: bool, mag: u64, hint: Option<&Type>) -> Result<Value> {
    let n = if neg {
        -i128::from(mag)
    } else {
        i128::from(mag)
    };
    let shown = if neg {
        format!("-{mag}")
    } else {
        mag.to_string()
    };
    match hint {
        Some(t) if t.is_int() => Value::int_of_type(t, n)
            .ok_or_else(|| err!(Overflow, "literal {shown} out of range for {t}")),
        Some(Type::F64) if mag <= 1 << 53 => Ok(Value::F64(n as f64)),
        Some(Type::F32) if mag <= 1 << 24 => Ok(Value::F32(n as f32)),
        Some(t @ (Type::F64 | Type::F32)) => Err(err!(
            Value,
            "literal {shown} cannot be represented exactly as {t}"
        )),
        _ => Value::int_of_type(&Type::I64, n).ok_or_else(|| {
            err!(
                Overflow,
                "literal {shown} out of range for int; annotate a wider type, e.g. x: u64 = {shown}"
            )
        }),
    }
}

fn float_literal(x: f64, hint: Option<&Type>) -> Value {
    match hint {
        Some(Type::F32) => Value::F32(x as f32),
        _ => Value::F64(x),
    }
}

fn normalize_index(idx: &Value, len: usize, what: &str) -> Result<usize> {
    let i = idx.as_i128().ok_or_else(|| {
        err!(
            Type,
            "{what} indices must be integers, not '{}'",
            idx.type_of()
        )
    })?;
    let len_i = len as i128;
    let j = if i < 0 { i + len_i } else { i };
    if (0..len_i).contains(&j) {
        Ok(j as usize)
    } else {
        Err(err!(Index, "{what} index {i} out of range (length {len})"))
    }
}

fn dict_key(d: &crate::value::Dict, idx: &Value) -> Result<Key> {
    if !d.key.accepts(&idx.type_of()) {
        return Err(err!(
            Type,
            "dict key must be '{}', not '{}'",
            d.key,
            idx.type_of()
        ));
    }
    Key::from_value(idx)
}

fn index(base: &Value, idx: &Value) -> Result<Value> {
    match base {
        Value::List(l) => Ok(l.items[normalize_index(idx, l.items.len(), "list")?].clone()),
        Value::Tuple(t) => Ok(t[normalize_index(idx, t.len(), "tuple")?].clone()),
        Value::Str(s) => {
            let n = s.chars().count();
            let i = normalize_index(idx, n, "str")?;
            Ok(Value::Str(s.chars().nth(i).unwrap().to_string().into()))
        }
        Value::Bytes(b) => Ok(Value::U8(b[normalize_index(idx, b.len(), "bytes")?])),
        Value::Dict(d) => {
            let k = dict_key(d, idx)?;
            d.map
                .get(&k)
                .cloned()
                .ok_or_else(|| err!(Key, "{}", idx.repr()))
        }
        other => Err(err!(Type, "'{}' is not subscriptable", other.type_of())),
    }
}

/// Write `rhs` (or `slot op rhs`) into `container[idxs[0]][idxs[1]]...`,
/// copying shared collections on write.
fn store(
    container: &mut Value,
    idxs: &[Value],
    op: Option<BinOp>,
    rhs: Value,
    max_len: usize,
) -> Result<()> {
    let (idx, rest) = idxs.split_first().expect("non-empty path");
    let combine = |old: Option<&Value>, rhs: Value| -> Result<Value> {
        match (op, old) {
            (None, _) => Ok(rhs),
            (Some(op), Some(old)) => ops::binary(op, old, &rhs, max_len),
            (Some(_), None) => Err(err!(Key, "{}", idx.repr())),
        }
    };
    match container {
        Value::List(l) => {
            let i = normalize_index(idx, l.items.len(), "list")?;
            let l = Rc::make_mut(l);
            let new = if rest.is_empty() {
                combine(Some(&l.items[i]), rhs)?
            } else {
                let mut child = l.items[i].clone();
                store(&mut child, rest, op, rhs, max_len)?;
                child
            };
            l.elem = l.elem.unify(&new.type_of()).ok_or_else(|| {
                err!(
                    Type,
                    "cannot store '{}' in 'list[{}]'",
                    new.type_of(),
                    l.elem
                )
            })?;
            l.items[i] = new;
            Ok(())
        }
        Value::Dict(d) => {
            let key = dict_key(d, idx)?;
            let d = Rc::make_mut(d);
            let new = if rest.is_empty() {
                combine(d.map.get(&key), rhs)?
            } else {
                let mut child = d
                    .map
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| err!(Key, "{}", idx.repr()))?;
                store(&mut child, rest, op, rhs, max_len)?;
                child
            };
            let val = d.val.unify(&new.type_of()).ok_or_else(|| {
                err!(
                    Type,
                    "cannot store '{}' in 'dict[{}, {}]'",
                    new.type_of(),
                    d.key,
                    d.val
                )
            })?;
            let key_ty = d.key.unify(&idx.type_of()).expect("checked by dict_key");
            d.val = val;
            d.key = key_ty;
            d.map.insert(key, new);
            Ok(())
        }
        Value::Tuple(_) | Value::Str(_) | Value::Bytes(_) => Err(err!(
            Type,
            "'{}' is immutable and does not support item assignment",
            container.type_of()
        )),
        other => Err(err!(
            Type,
            "'{}' does not support item assignment",
            other.type_of()
        )),
    }
}

/// The items produced by iterating a value (used by `map`).
fn iterate(v: &Value) -> Result<Vec<Value>> {
    Ok(match v {
        Value::List(l) => l.items.clone(),
        Value::Set(s) => s.items.iter().map(Key::to_value).collect(),
        Value::Dict(d) => d.map.keys().map(Key::to_value).collect(),
        Value::Str(s) => s
            .chars()
            .map(|c| Value::Str(c.to_string().into()))
            .collect(),
        Value::Bytes(b) => b.iter().map(|&x| Value::U8(x)).collect(),
        Value::Tuple(items) => {
            if v.type_of().iter_elem().is_none() {
                return Err(err!(
                    Type,
                    "cannot iterate over a mixed-type tuple '{}'",
                    v.type_of()
                ));
            }
            items.to_vec()
        }
        other => Err(err!(Type, "'{}' is not iterable", other.type_of()))?,
    })
}

//! Tree-walking evaluator and the embeddable [`Interpreter`] object.
//!
//! # Scoping
//!
//! * Every function call and every `if`/`elif`/`else`/`while`/`for` body
//!   opens a new scope for declarations.
//! * `x: T = v` always declares `x` in the current scope (shadowing).
//! * `x = v` redeclares `x` if it lives in the current scope (its type may
//!   change), writes it if it lives in an enclosing scope of the same
//!   function (its type must not change), and declares it otherwise.
//! * A `def`/`lambda` copies the outer variables its body uses when it is
//!   declared. Afterwards it never sees outer changes, and its writes only
//!   affect its own copy, except for lock variables (`rwlock[T]`,
//!   `rlock[T]`, `wlock[T]`), which are shared, with permissions checked
//!   when the function is declared.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{Read as _, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};

use crate::ast::{
    BinOp, CmpOp, Expr, ExprKind, FreeVars, FuncDef, LambdaDef, LogicOp, READ_WRITE_METHODS, Stmt,
    StmtKind, UnOp, is_mutating_method, target_path,
};
use crate::error::{ErrorKind, Result, err};
use crate::native::{IntoNativeFn, NativeFn};
use crate::ops;
use crate::types::{FnSig, LockKind, MAX_TYPE_DEPTH, Type};
use crate::value::{Dict, FileHandle, IterView, Key, List, Set, Value, check_depth};

const STACK_RED_ZONE: usize = 128 * 1024;
const STACK_SEGMENT: usize = 2 * 1024 * 1024;

/// Host-controlled limits and capabilities.
#[derive(Debug, Clone)]
pub struct Config {
    /// Whether scripts may call `open`. Off by default: a script can only
    /// touch the filesystem if the embedder opts in.
    pub allow_fs: bool,
    /// Maximum evaluation nesting (expression depth plus call depth). Bounds
    /// recursion so runaway scripts error instead of exhausting memory.
    pub max_eval_depth: usize,
    /// Maximum length in bytes of any string/bytes value produced by
    /// concatenation or file reads.
    pub max_str_len: usize,
    /// Maximum number of steps (statements, loop iterations and iterator
    /// items) per `run`/`eval`/`call`. `None` for no limit.
    pub max_steps: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            allow_fs: false,
            max_eval_depth: 400,
            max_str_len: 16 * 1024 * 1024,
            max_steps: Some(10_000_000),
        }
    }
}

// ------------------------------------------------------------------ scopes

/// Storage behind a lock-annotated variable.
pub(crate) struct Cell {
    ty: Type,
    kind: LockKind,
    value: RwLock<Value>,
}

impl Cell {
    fn get(&self) -> Value {
        self.value.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn guard(&self) -> std::sync::RwLockWriteGuard<'_, Value> {
        self.value.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// What a scope may do with a lock variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Full,
    Read,
    Write,
}

impl Access {
    fn can_read(self) -> bool {
        self != Access::Write
    }

    fn can_write(self) -> bool {
        self != Access::Read
    }

    /// The access a nested function gets to a lock it captures.
    fn narrow(self, kind: LockKind) -> Access {
        match (self, kind) {
            (Access::Full, LockKind::ReadWrite) => Access::Full,
            (Access::Full, LockKind::Read) => Access::Read,
            (Access::Full, LockKind::Write) => Access::Write,
            (outer, _) => outer,
        }
    }
}

#[derive(Clone)]
enum Slot {
    /// A plain variable and the type it was declared with, if annotated.
    Val(Value, Option<Type>),
    Cell(Arc<Cell>, Access),
}

impl Slot {
    fn val(v: Value) -> Slot {
        Slot::Val(v, None)
    }
}

type Scope = HashMap<String, Slot>;

/// The scope stack of one function call (or of the top level).
struct Env {
    scopes: Vec<Scope>,
}

impl Env {
    fn new(base: Scope) -> Env {
        Env { scopes: vec![base] }
    }

    /// Innermost scope index holding `name`, and its slot.
    fn find(&self, name: &str) -> Option<(usize, &Slot)> {
        self.scopes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, s)| s.get(name).map(|slot| (i, slot)))
    }

    fn top(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("never empty")
    }

    fn is_top(&self, i: usize) -> bool {
        i + 1 == self.scopes.len()
    }
}

fn read_slot(name: &str, slot: &Slot) -> Result<Value> {
    match slot {
        Slot::Val(v, _) => Ok(v.clone()),
        Slot::Cell(c, access) if access.can_read() => Ok(c.get()),
        Slot::Cell(c, _) => Err(err!(
            Permission,
            "'{name}' is a {}[{}] and is write-only here",
            c.kind.name(),
            c.ty
        )),
    }
}

fn require_write(name: &str, slot: &Slot, also_read: bool) -> Result<()> {
    if let Slot::Cell(c, access) = slot {
        let ok = access.can_write() && (!also_read || access.can_read());
        if !ok {
            let what = if access.can_write() {
                "write-only"
            } else {
                "read-only"
            };
            return Err(err!(
                Permission,
                "'{name}' is a {}[{}] and is {what} here",
                c.kind.name(),
                c.ty
            ));
        }
    }
    Ok(())
}

/// A lock variable shared between the host and scripts (see
/// [`Interpreter::share`]). Cloning shares the same storage.
#[derive(Clone)]
pub struct Shared(Arc<Cell>);

impl Shared {
    pub fn get(&self) -> Value {
        self.0.get()
    }

    /// Replace the value. It must fit the declared type.
    pub fn set(&self, value: impl Into<Value>) -> Result<()> {
        let mut v = value.into();
        if !self.0.ty.accepts(&v.type_of()) {
            return Err(err!(
                Type,
                "value of type '{}' does not fit '{}'",
                v.type_of(),
                self.0.ty
            ));
        }
        v.refine(&self.0.ty);
        check_no_cycle(&self.0, "shared lock", std::slice::from_ref(&v))?;
        *self.0.guard() = v;
        Ok(())
    }

    pub fn ty(&self) -> &Type {
        &self.0.ty
    }
}

// --------------------------------------------------------------- functions

/// A user-defined function or lambda.
pub struct Function {
    pub name: String,
    params: Vec<(String, Option<Type>)>,
    ret: Option<Type>,
    body: Body,
    /// The outer variables the body uses, copied (or, for locks, shared)
    /// when the function was declared.
    captured: Arc<Vec<(String, Slot)>>,
}

enum Body {
    Block(Arc<FuncDef>),
    Lambda(Arc<LambdaDef>),
}

const LAMBDA: &str = "<lambda>";

impl Function {
    /// `Callable[[A, ...], R]`; unannotated parts are unknown.
    pub fn signature(&self) -> Type {
        Type::Function(Some(Box::new(FnSig {
            params: self
                .params
                .iter()
                .map(|(_, t)| t.clone().unwrap_or(Type::Unknown))
                .collect(),
            ret: self.ret.clone().unwrap_or(Type::Unknown),
        })))
    }
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
    Range,
    Len,
    List,
    Set,
    /// The `json` namespace; `json.loads` / `json.dumps` are attributes.
    Json,
    JsonLoads,
    JsonDumps,
    /// A type used as a conversion function, e.g. `u8(x)` or `Path("a")`.
    Convert(Type),
}

impl Builtin {
    fn lookup(name: &str) -> Option<Builtin> {
        Some(match name {
            "print" => Builtin::Print,
            "map" => Builtin::Map,
            "open" => Builtin::Open,
            "range" => Builtin::Range,
            "len" => Builtin::Len,
            "list" => Builtin::List,
            "set" => Builtin::Set,
            "json" => Builtin::Json,
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
            Builtin::Range => "range".into(),
            Builtin::Len => "len".into(),
            Builtin::List => "list".into(),
            Builtin::Set => "set".into(),
            Builtin::Json => "json".into(),
            Builtin::JsonLoads => "json.loads".into(),
            Builtin::JsonDumps => "json.dumps".into(),
            Builtin::Convert(t) => t.to_string(),
        }
    }

    /// Declared return type, where it doesn't depend on the arguments.
    pub fn ret_type(&self) -> Type {
        match self {
            Builtin::Print => Type::optional(Type::Bool),
            Builtin::Len => Type::I64,
            Builtin::JsonDumps => Type::Str,
            Builtin::JsonLoads => Type::Any,
            _ => Type::Unknown,
        }
    }
}

/// Return type of a callable, if declared.
fn ret_type_of(f: &Value) -> Type {
    match f {
        Value::Function(func) => func.ret.clone().unwrap_or(Type::Unknown),
        Value::Native(n) => n.ret().clone(),
        Value::Builtin(b) => b.ret_type(),
        _ => Type::Unknown,
    }
}

enum Flow {
    Next(Option<Value>),
    Return(Value),
    Break,
    Continue,
}

/// Whether a chunk of source is ready to run. Useful for building REPLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputStatus {
    Complete,
    /// More lines are needed (open bracket, empty block body, ...).
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

/// An iterator in progress. Holds snapshots, so it is unaffected by
/// mutation of the source during the loop.
enum IterState {
    List(Arc<List>, usize),
    Tuple(Arc<[Value]>, usize),
    Set(Arc<Set>, usize),
    Keys(Arc<Dict>, usize),
    Values(Arc<Dict>, usize),
    Items(Arc<Dict>, usize),
    Str(Arc<str>, usize),
    Bytes(Arc<[u8]>, usize),
    Range {
        cur: i128,
        stop: i128,
        step: i128,
        ty: Type,
    },
    Map {
        f: Value,
        inner: Box<IterState>,
    },
}

/// An embeddable femtopython interpreter. Global variables persist across
/// calls to [`run`](Interpreter::run) and [`eval`](Interpreter::eval).
/// `Interpreter` is `Send`, so it can be moved to (or created on) a worker
/// thread.
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
    globals: Scope,
    out: Box<dyn Write + Send>,
    depth: usize,
    steps: u64,
    interrupt: Arc<AtomicBool>,
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Interpreter>();
};

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
            steps: 0,
            interrupt: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Redirect the output of `print`.
    pub fn set_output(&mut self, out: impl Write + Send + 'static) {
        self.out = Box::new(out);
    }

    /// A flag that, when set (from any thread), makes the running script
    /// stop with an [`ErrorKind::Interrupted`] error at its next step.
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupt.clone()
    }

    fn reset_budget(&mut self) {
        self.depth = 0;
        self.steps = 0;
    }

    /// Run a program (any number of statements). Returns the value of the
    /// final statement if it is an expression, otherwise `None`. If an error
    /// occurs, the effects of statements before it are kept.
    pub fn run(&mut self, src: &str) -> Result<Value> {
        let program = crate::parser::parse_program(src)?;
        crate::check::check_program(&program)?;
        self.reset_budget();
        self.with_globals(|this, env| {
            let mut last = Value::None;
            for stmt in &program {
                last = match this.exec_stmt(stmt, env, None)? {
                    Flow::Next(v) => v.unwrap_or(Value::None),
                    _ => unreachable!("checker rejects top-level return/break/continue"),
                };
            }
            Ok(last)
        })
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
        self.reset_budget();
        self.with_globals(|this, env| this.eval_expr(expr, env, None))
    }

    fn with_globals<R>(&mut self, f: impl FnOnce(&mut Self, &mut Env) -> Result<R>) -> Result<R> {
        let mut env = Env::new(std::mem::take(&mut self.globals));
        let r = f(self, &mut env);
        env.scopes.truncate(1);
        self.globals = env.scopes.pop().expect("base scope");
        r
    }

    /// Read a global variable (including lock variables).
    pub fn get(&self, name: &str) -> Option<Value> {
        self.globals.get(name).and_then(|s| read_slot(name, s).ok())
    }

    /// Declare (or redeclare) a global variable.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.globals.insert(name.into(), Slot::val(value.into()));
    }

    /// Declare a global lock variable, as if by `name: kind[ty] = value`,
    /// and return a handle the host can use to read and write it, even while
    /// a script runs on another thread.
    pub fn share(
        &mut self,
        name: impl Into<String>,
        kind: LockKind,
        ty: Type,
        value: impl Into<Value>,
    ) -> Result<Shared> {
        let shared = Shared(Arc::new(Cell {
            ty,
            kind,
            value: RwLock::new(Value::None),
        }));
        shared.set(value)?;
        self.globals
            .insert(name.into(), Slot::Cell(shared.0.clone(), Access::Full));
        Ok(shared)
    }

    pub fn globals(&self) -> impl Iterator<Item = (&str, Value)> {
        self.globals
            .iter()
            .filter_map(|(k, s)| Some((k.as_str(), read_slot(k, s).ok()?)))
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
        self.globals
            .insert(name, Slot::val(Value::Native(Arc::new(f))));
    }

    /// Expose an unchecked function that receives the raw argument values,
    /// for variadic or dynamically-typed host functions.
    pub fn register_raw_fn(
        &mut self,
        name: impl Into<String>,
        func: impl Fn(&[Value]) -> Result<Value> + Send + Sync + 'static,
    ) {
        let name = name.into();
        let f = NativeFn::raw(name.clone(), func);
        self.globals
            .insert(name, Slot::val(Value::Native(Arc::new(f))));
    }

    /// Call a script function (e.g. one fetched with [`get`](Self::get))
    /// from the host.
    pub fn call(&mut self, f: &Value, args: Vec<Value>) -> Result<Value> {
        self.reset_budget();
        self.call_value(f, args)
    }

    fn step(&mut self) -> Result<()> {
        self.steps += 1;
        if let Some(max) = self.config.max_steps
            && self.steps > max
        {
            return Err(err!(Limit, "step limit ({max}) exceeded"));
        }
        if self.interrupt.swap(false, AtomicOrdering::Relaxed) {
            return Err(err!(Interrupted, "interrupted"));
        }
        Ok(())
    }

    // ------------------------------------------------------------ statements

    fn exec_stmts(&mut self, stmts: &[Stmt], env: &mut Env, ret: Option<&Type>) -> Result<Flow> {
        for s in stmts {
            match self.exec_stmt(s, env, ret)? {
                Flow::Next(_) => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Next(None))
    }

    /// Run `stmts` in a new scope pre-populated with `pre`.
    fn exec_block(
        &mut self,
        stmts: &[Stmt],
        env: &mut Env,
        ret: Option<&Type>,
        pre: Scope,
    ) -> Result<Flow> {
        env.scopes.push(pre);
        let r = self.exec_stmts(stmts, env, ret);
        env.scopes.pop();
        r
    }

    fn exec_stmt(&mut self, stmt: &Stmt, env: &mut Env, ret: Option<&Type>) -> Result<Flow> {
        self.step().map_err(|e| e.at(stmt.span))?;
        Ok(match &stmt.kind {
            StmtKind::Expr(e) => Flow::Next(Some(self.eval_expr(e, env, None)?)),
            StmtKind::Declare { name, ty, value } => {
                self.declare(env, name, ty.as_ref(), value)?;
                Flow::Next(None)
            }
            StmtKind::Assign { target, value } => {
                self.assign(env, target, None, value)?;
                Flow::Next(None)
            }
            StmtKind::AugAssign { target, op, value } => {
                self.assign(env, target, Some(*op), value)?;
                Flow::Next(None)
            }
            StmtKind::Return(e) => Flow::Return(match e {
                Some(e) => self.eval_expr(e, env, ret)?,
                None => Value::None,
            }),
            StmtKind::Pass => Flow::Next(None),
            StmtKind::Break => Flow::Break,
            StmtKind::Continue => Flow::Continue,
            StmtKind::Def(def) => {
                let captured = capture(env, &def.free, &def.name).map_err(|e| e.at(stmt.span))?;
                let f = Function {
                    name: def.name.clone(),
                    params: def
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.ty.clone()))
                        .collect(),
                    ret: def.ret.clone(),
                    body: Body::Block(def.clone()),
                    captured,
                };
                env.top()
                    .insert(def.name.clone(), Slot::val(Value::Function(Arc::new(f))));
                Flow::Next(None)
            }
            StmtKind::If { branches, orelse } => {
                for (cond, body) in branches {
                    if self.eval_bool(cond, env, "an if condition")? {
                        return self.exec_block(body, env, ret, Scope::new());
                    }
                }
                match orelse {
                    Some(body) => self.exec_block(body, env, ret, Scope::new())?,
                    None => Flow::Next(None),
                }
            }
            StmtKind::While { cond, body } => {
                while self.eval_bool(cond, env, "a while condition")? {
                    match self.exec_block(body, env, ret, Scope::new())? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Next(_) | Flow::Continue => {}
                    }
                    self.step().map_err(|e| e.at(stmt.span))?;
                }
                Flow::Next(None)
            }
            StmtKind::For {
                targets,
                iter,
                body,
            } => {
                let it = self.eval_expr(iter, env, None)?;
                let mut state = start_iter(&it).map_err(|e| e.at(iter.span))?;
                while let Some(item) = self.next_item(&mut state).map_err(|e| e.at(iter.span))? {
                    let pre = bind_targets(targets, item).map_err(|e| e.at(stmt.span))?;
                    match self.exec_block(body, env, ret, pre)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Next(_) | Flow::Continue => {}
                    }
                }
                Flow::Next(None)
            }
        })
    }

    fn declare(
        &mut self,
        env: &mut Env,
        name: &str,
        ty: Option<&Type>,
        value: &Expr,
    ) -> Result<()> {
        let checked = |v: &mut Value, ty: &Type| {
            if !ty.accepts(&v.type_of()) {
                return Err(err!(
                    Type,
                    "'{name}' is declared as '{ty}' but the value has type '{}'",
                    v.type_of()
                )
                .at(value.span));
            }
            v.refine(ty);
            Ok(())
        };
        match ty {
            Some(Type::Lock(kind, t)) => {
                let mut v = self.eval_expr(value, env, Some(t))?;
                checked(&mut v, t)?;
                let cell = Cell {
                    ty: (**t).clone(),
                    kind: *kind,
                    value: RwLock::new(v),
                };
                env.top()
                    .insert(name.to_owned(), Slot::Cell(Arc::new(cell), Access::Full));
            }
            Some(t) => {
                let mut v = self.eval_expr(value, env, Some(t))?;
                checked(&mut v, t)?;
                env.top()
                    .insert(name.to_owned(), Slot::Val(v, Some(t.clone())));
            }
            None => {
                let found = env.find(name).map(|(i, s)| (i, s.clone()));
                match found {
                    Some((_, Slot::Cell(c, access))) => {
                        require_write(name, &Slot::Cell(c.clone(), access), false)
                            .map_err(|e| e.at(value.span))?;
                        let mut v = self.eval_expr(value, env, Some(&c.ty))?;
                        checked(&mut v, &c.ty)?;
                        check_no_cycle(&c, name, std::slice::from_ref(&v))
                            .map_err(|e| e.at(value.span))?;
                        *c.guard() = v;
                    }
                    Some((i, Slot::Val(old, declared))) => {
                        let ty = declared.clone().unwrap_or_else(|| old.type_of());
                        let mut v = self.eval_expr(value, env, Some(&ty))?;
                        if ty.accepts(&v.type_of()) {
                            // An assignment: keep the declared type.
                            v.refine(&ty);
                            env.scopes[i].insert(name.to_owned(), Slot::Val(v, declared));
                        } else if env.is_top(i) {
                            // A redeclaration with a new type.
                            env.top().insert(name.to_owned(), Slot::val(v));
                        } else {
                            return Err(err!(
                                Type,
                                "cannot assign '{}' to '{name}', which has type '{ty}' in an \
                                 enclosing block (write `{name}: T = ...` to declare a new variable)",
                                v.type_of()
                            )
                            .at(value.span));
                        }
                    }
                    None => {
                        let v = self.eval_expr(value, env, None)?;
                        env.top().insert(name.to_owned(), Slot::val(v));
                    }
                }
            }
        }
        Ok(())
    }

    /// `target[i]... = value` and `target[i]... op= value`.
    fn assign(
        &mut self,
        env: &mut Env,
        target: &Expr,
        op: Option<BinOp>,
        value: &Expr,
    ) -> Result<()> {
        let (root, path) = target_path(target).expect("parser only builds valid targets");
        let mut idxs = Vec::with_capacity(path.len());
        for e in &path {
            idxs.push(self.eval_expr(e, env, None)?);
        }
        let (scope_i, slot) = env
            .find(root)
            .map(|(i, s)| (i, s.clone()))
            .ok_or_else(|| err!(Name, "name '{root}' is not defined").at(target.span))?;
        require_write(root, &slot, op.is_some()).map_err(|e| e.at(target.span))?;
        let root_ty = match &slot {
            Slot::Val(v, declared) => declared.clone().unwrap_or_else(|| v.type_of()),
            Slot::Cell(c, _) => c.ty.clone(),
        };
        let hint = match op {
            Some(op) if ops::takes_int_amount(op) && slot_type(&root_ty, &idxs).is_int() => None,
            _ => Some(slot_type(&root_ty, &idxs)).filter(|t| *t != Type::Unknown),
        };
        let rhs = self.eval_expr(value, env, hint.as_ref())?;
        if let Slot::Cell(c, _) = &slot {
            check_no_cycle(c, root, std::slice::from_ref(&rhs)).map_err(|e| e.at(target.span))?;
        }
        let max_len = self.config.max_str_len;
        let span = target.span;
        let apply = |v: &mut Value| -> Result<()> {
            match idxs.split_last() {
                None => {
                    let op = op.expect("plain name assignment is a declaration");
                    *v = ops::binary(op, v, &rhs, max_len)?;
                    Ok(())
                }
                Some((last, parents)) => {
                    with_place(v, parents, |c| set_item(c, last, op, rhs, max_len))
                }
            }
        };
        match slot {
            Slot::Val(..) => {
                let Some(Slot::Val(v, _)) = env.scopes[scope_i].get_mut(root) else {
                    unreachable!()
                };
                apply(v).map_err(|e| e.at(span))
            }
            // The lock's value was refined to its declared type when it was
            // created, so element-type checks in `set_item` keep it in that
            // type, and same-type operators can't change it. Every check in
            // `apply` runs before it mutates anything.
            Slot::Cell(c, _) => apply(&mut c.guard()).map_err(|e| e.at(span)),
        }
    }

    // ----------------------------------------------------------- expressions

    /// Evaluate `e`. `hint` is the type the context expects; it only affects
    /// numeric literals, which adopt it (as Rust's integer literals do).
    fn eval_expr(&mut self, e: &Expr, env: &mut Env, hint: Option<&Type>) -> Result<Value> {
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
                self.eval_inner(e, env, hint)
            })
        };
        self.depth -= 1;
        r.map_err(|err| err.at(e.span))
    }

    fn eval_inner(&mut self, e: &Expr, env: &mut Env, hint: Option<&Type>) -> Result<Value> {
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
            ExprKind::Name(n) => lookup(env, n)?,
            ExprKind::List(items) => {
                let elem_hint = match hint {
                    Some(Type::List(t)) => Some(&**t),
                    _ => None,
                };
                let refs: Vec<&Expr> = items.iter().collect();
                let vals = self.eval_group(&refs, env, elem_hint)?;
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
                let vals = self.eval_group(&refs, env, elem_hint)?;
                Value::new_set(Type::Unknown, vals)?
            }
            ExprKind::Dict(entries) => {
                let (kh, vh) = match hint {
                    Some(Type::Dict(k, v)) => (Some(&**k), Some(&**v)),
                    _ => (None, None),
                };
                let keys: Vec<&Expr> = entries.iter().map(|(k, _)| k).collect();
                let vals: Vec<&Expr> = entries.iter().map(|(_, v)| v).collect();
                let keys = self.eval_group(&keys, env, kh)?;
                // Values are heterogeneous (JSON-style), so each literal is
                // typed on its own rather than unified with its siblings.
                let vals = vals
                    .into_iter()
                    .map(|v| self.eval_expr(v, env, vh))
                    .collect::<Result<Vec<_>>>()?;
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
                    vals.push(self.eval_expr(item, env, hints.map(|h| &h[i]))?);
                }
                Value::new_tuple(vals)?
            }
            ExprKind::Unary(op, inner) => {
                let h = if *op == UnOp::Not { None } else { hint };
                let v = self.eval_expr(inner, env, h)?;
                ops::unary(*op, &v)?
            }
            ExprKind::Binary(op, a, b)
                if ops::takes_int_amount(*op)
                    && !(a.is_numeric_literal() && b.is_numeric_literal()) =>
            {
                let a = self.eval_expr(a, env, hint)?;
                // Integer shift amounts and exponents may be any integer
                // type; a float base needs a float exponent.
                let b_hint = a.type_of();
                let b = self.eval_expr(b, env, b_hint.is_float().then_some(&b_hint))?;
                ops::binary(*op, &a, &b, self.config.max_str_len)?
            }
            ExprKind::Binary(op, a, b) => {
                let [a, b] = self.eval_pair(a, b, env, hint)?;
                ops::binary(*op, &a, &b, self.config.max_str_len)?
            }
            ExprKind::Compare(op @ (CmpOp::In | CmpOp::NotIn), item, container) => {
                let (item, container) = if item.is_numeric_literal() {
                    let c = self.eval_expr(container, env, None)?;
                    let elem = c.type_of().iter_elem();
                    (self.eval_expr(item, env, elem.as_ref())?, c)
                } else {
                    let i = self.eval_expr(item, env, None)?;
                    let t = Box::new(i.type_of());
                    let hint = match &container.kind {
                        ExprKind::List(_) => Some(Type::List(t)),
                        ExprKind::Set(_) => Some(Type::Set(t)),
                        ExprKind::Dict(_) => Some(Type::Dict(t, Box::new(Type::Unknown))),
                        _ => None,
                    };
                    (i, self.eval_expr(container, env, hint.as_ref())?)
                };
                let found = match ops::contains(&container, &item)? {
                    Some(b) => b,
                    None => self.iter_contains(&container, &item)?,
                };
                Value::Bool(found == (*op == CmpOp::In))
            }
            ExprKind::Compare(op, a, b) => {
                let [a, b] = self.eval_pair(a, b, env, None)?;
                Value::Bool(ops::compare(*op, &a, &b)?)
            }
            ExprKind::IsNone(inner, negated) => {
                let v = self.eval_expr(inner, env, None)?;
                Value::Bool(matches!(v, Value::None) != *negated)
            }
            ExprKind::Logic(op, a, b) => {
                let a_val = self.eval_bool(a, env, "and/or")?;
                let short = match op {
                    LogicOp::And => !a_val,
                    LogicOp::Or => a_val,
                };
                if short {
                    Value::Bool(a_val)
                } else {
                    Value::Bool(self.eval_bool(b, env, "and/or")?)
                }
            }
            ExprKind::IfExp { body, cond, orelse } => {
                if self.eval_bool(cond, env, "a conditional expression")? {
                    self.eval_expr(body, env, hint)?
                } else {
                    self.eval_expr(orelse, env, hint)?
                }
            }
            ExprKind::Call(callee, args) => return self.eval_call(callee, args, env, hint),
            ExprKind::Index(base, idx) => {
                let base = self.eval_expr(base, env, None)?;
                let idx_hint = match base.type_of() {
                    Type::Dict(k, _) => Some(*k),
                    _ => None,
                };
                let idx = self.eval_expr(idx, env, idx_hint.as_ref())?;
                index(&base, &idx)?
            }
            ExprKind::Attr(obj, name) => {
                let obj = self.eval_expr(obj, env, None)?;
                if let Value::Builtin(Builtin::Json) = obj {
                    return module_attr(&obj, name);
                }
                return Err(err!(
                    Type,
                    "'{}' attribute '{name}' must be called as a method",
                    obj.type_of()
                ));
            }
            ExprKind::Lambda(def) => Value::Function(Arc::new(Function {
                name: LAMBDA.into(),
                params: def.params.iter().map(|p| (p.clone(), None)).collect(),
                ret: None,
                body: Body::Lambda(def.clone()),
                captured: capture(env, &def.free, LAMBDA)?,
            })),
        })
    }

    /// `hint` is the type the context expects of the call's result. It lets
    /// `json.dumps` and `file.read` produce `bytes` instead of `str`.
    fn eval_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        env: &mut Env,
        hint: Option<&Type>,
    ) -> Result<Value> {
        let f = if let ExprKind::Attr(obj, method) = &callee.kind {
            if is_mutating_method(method) {
                return self.call_mutating_method(obj, method, args, env);
            }
            let obj = self.eval_expr(obj, env, None)?;
            if let Value::Builtin(Builtin::Json) = obj {
                module_attr(&obj, method)?
            } else {
                let arg_hint = match (&obj, method.as_str()) {
                    (Value::Dict(d), "get") => Some(d.key.clone()),
                    _ => None,
                };
                let mut vals = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    let h = if i == 0 { arg_hint.as_ref() } else { None };
                    vals.push(self.eval_expr(a, env, h)?);
                }
                return self.call_method(&obj, method, vals, hint);
            }
        } else {
            self.eval_expr(callee, env, None)?
        };
        let vals = if matches!(f, Value::Builtin(Builtin::Range)) {
            let refs: Vec<&Expr> = args.iter().collect();
            self.eval_group(&refs, env, None)?
        } else {
            let hints = param_hints(&f);
            let mut vals = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let h = hints.get(i).and_then(Option::as_ref);
                let v = match self.eval_expr(a, env, h) {
                    // `u8(300)`: a literal that doesn't fit the target type
                    // stays an `int`, so the conversion yields `None`.
                    Err(e)
                        if e.kind == ErrorKind::Overflow
                            && a.is_numeric_literal()
                            && matches!(f, Value::Builtin(Builtin::Convert(_))) =>
                    {
                        self.eval_expr(a, env, None)?
                    }
                    r => r?,
                };
                vals.push(v);
            }
            vals
        };
        let out = self.call_value(&f, vals)?;
        Ok(match (&f, out) {
            (Value::Builtin(Builtin::JsonDumps), Value::Str(s)) if wants_bytes(hint) => {
                Value::Bytes(s.as_bytes().into())
            }
            (_, out) => out,
        })
    }

    fn eval_bool(&mut self, e: &Expr, env: &mut Env, what: &str) -> Result<bool> {
        match self.eval_expr(e, env, None)? {
            Value::Bool(b) => Ok(b),
            other => Err(err!(
                Type,
                "{what} must be bool, got '{}' (there is no implicit truthiness)",
                other.type_of()
            )
            .at(e.span)),
        }
    }

    fn eval_pair(
        &mut self,
        a: &Expr,
        b: &Expr,
        env: &mut Env,
        hint: Option<&Type>,
    ) -> Result<[Value; 2]> {
        let v = self.eval_group(&[a, b], env, hint)?;
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
        env: &mut Env,
        hint: Option<&Type>,
    ) -> Result<Vec<Value>> {
        let mut out: Vec<Option<Value>> = vec![None; exprs.len()];
        let mut found: Option<Type> = None;
        for (i, e) in exprs.iter().enumerate() {
            if !e.is_numeric_literal() {
                let v = self.eval_expr(e, env, hint)?;
                if found.is_none() && (v.type_of().is_int() || v.type_of().is_float()) {
                    found = Some(v.type_of());
                }
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
                out[i] = Some(self.eval_expr(e, env, lit_ty.as_ref())?);
            }
        }
        Ok(out.into_iter().map(Option::unwrap).collect())
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

    fn call_function(&mut self, func: &Arc<Function>, args: Vec<Value>) -> Result<Value> {
        if args.len() != func.params.len() {
            return Err(err!(
                Type,
                "{}() takes {} argument(s) but {} were given",
                func.name,
                func.params.len(),
                args.len()
            ));
        }
        let mut scope: Scope = func.captured.iter().cloned().collect();
        if func.name != LAMBDA {
            scope.insert(func.name.clone(), Slot::val(Value::Function(func.clone())));
        }
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
            scope.insert(name.clone(), Slot::Val(arg, ty.clone()));
        }
        let mut env = Env::new(scope);
        let mut result = match &func.body {
            Body::Lambda(l) => self.eval_expr(&l.body, &mut env, None)?,
            Body::Block(def) => match self.exec_stmts(&def.body, &mut env, func.ret.as_ref())? {
                Flow::Return(v) => v,
                Flow::Next(_) => Value::None,
                Flow::Break | Flow::Continue => unreachable!("checker rejects stray break"),
            },
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
        let arity = |range: std::ops::RangeInclusive<usize>| {
            if range.contains(&args.len()) {
                Ok(())
            } else {
                let want = if range.start() == range.end() {
                    range.start().to_string()
                } else {
                    format!("{} to {}", range.start(), range.end())
                };
                Err(err!(
                    Type,
                    "{}() takes {want} argument(s) but {} were given",
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
                arity(2..=2)?;
                let mut args = args.into_iter();
                let (f, src) = (args.next().unwrap(), args.next().unwrap());
                if !f.is_callable() {
                    return Err(err!(
                        Type,
                        "map() first argument must be a function, not '{}'",
                        f.type_of()
                    ));
                }
                start_iter(&src)?;
                let mut depth = 1;
                let mut cur = &src;
                while let Value::Iter(v) = cur
                    && let IterView::Map { src, .. } = &**v
                {
                    depth += 1;
                    cur = src;
                }
                if depth > MAX_TYPE_DEPTH {
                    return Err(err!(
                        Limit,
                        "map() nesting exceeds the maximum depth of {MAX_TYPE_DEPTH}"
                    ));
                }
                let elem = ret_type_of(&f);
                Ok(Value::Iter(Arc::new(IterView::Map { f, src, elem })))
            }
            Builtin::Range => {
                arity(1..=3)?;
                let ty = args[0].type_of();
                let mut bounds = Vec::with_capacity(3);
                for a in &args {
                    match a.as_i128() {
                        Some(n) if a.type_of() == ty => bounds.push(n),
                        _ => {
                            return Err(err!(
                                Type,
                                "range() arguments must be integers of one type, got '{}' and '{}'",
                                ty,
                                a.type_of()
                            ));
                        }
                    }
                }
                let (start, stop, step) = match bounds[..] {
                    [stop] => (0, stop, 1),
                    [start, stop] => (start, stop, 1),
                    [start, stop, step] => (start, stop, step),
                    _ => unreachable!(),
                };
                if step == 0 {
                    return Err(err!(Value, "range() step must not be zero"));
                }
                Ok(Value::Iter(Arc::new(IterView::Range {
                    start,
                    stop,
                    step,
                    ty,
                })))
            }
            Builtin::Len => {
                arity(1..=1)?;
                let n = match &args[0] {
                    Value::List(l) => l.items.len(),
                    Value::Dict(d) => d.map.len(),
                    Value::Set(s) => s.items.len(),
                    Value::Str(s) => s.chars().count(),
                    Value::Bytes(b) => b.len(),
                    Value::Tuple(t) => t.len(),
                    Value::Iter(v) if v.known_len().is_some() => v.known_len().unwrap(),
                    other => {
                        return Err(err!(
                            Type,
                            "object of type '{}' has no len()",
                            other.type_of()
                        ));
                    }
                };
                Ok(Value::I64(n as i64))
            }
            Builtin::List | Builtin::Set => {
                arity(0..=1)?;
                let (elem, items) = match args.first() {
                    None => (Type::Unknown, Vec::new()),
                    Some(src) => {
                        let elem = src.type_of().iter_elem().unwrap_or(Type::Unknown);
                        (elem, self.collect(src)?)
                    }
                };
                if *b == Builtin::List {
                    Value::new_list(elem, items)
                } else {
                    Value::new_set(elem, items)
                }
            }
            Builtin::Open => {
                arity(1..=1)?;
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
                Ok(Value::File(Arc::new(Mutex::new(FileHandle { path, file }))))
            }
            Builtin::Convert(t) => {
                arity(1..=1)?;
                ops::convert(t, &args[0])
            }
            Builtin::Json => Err(err!(Type, "module 'json' is not callable")),
            Builtin::JsonLoads => {
                arity(1..=1)?;
                match &args[0] {
                    Value::Str(s) => crate::json::loads(s),
                    Value::Bytes(b) => crate::json::loads(
                        std::str::from_utf8(b)
                            .map_err(|e| err!(Value, "JSON input is not valid UTF-8: {e}"))?,
                    ),
                    other => Err(err!(
                        Type,
                        "json.loads() expects str or bytes, not '{}'",
                        other.type_of()
                    )),
                }
            }
            Builtin::JsonDumps => {
                arity(1..=1)?;
                Ok(Value::Str(
                    crate::json::dumps(&args[0], self.config.max_str_len)?.into(),
                ))
            }
        }
    }

    fn call_method(
        &mut self,
        obj: &Value,
        method: &str,
        args: Vec<Value>,
        hint: Option<&Type>,
    ) -> Result<Value> {
        let arity = |range: std::ops::RangeInclusive<usize>| {
            if range.contains(&args.len()) {
                Ok(())
            } else {
                Err(err!(Type, "{method}() got {} argument(s)", args.len()))
            }
        };
        let view = |v: IterView| Ok(Value::Iter(Arc::new(v)));
        match (obj, method) {
            (Value::Dict(d), "keys") => {
                arity(0..=0)?;
                view(IterView::Keys(d.clone()))
            }
            (Value::Dict(d), "values") => {
                arity(0..=0)?;
                view(IterView::Values(d.clone()))
            }
            (Value::Dict(d), "items") => {
                arity(0..=0)?;
                view(IterView::Items(d.clone()))
            }
            (Value::Dict(d), "get") => {
                arity(1..=2)?;
                let k = dict_key(d, &args[0])?;
                Ok(d.map
                    .get(&k)
                    .cloned()
                    .unwrap_or_else(|| args.get(1).cloned().unwrap_or(Value::None)))
            }
            (Value::File(f), "read") => {
                arity(0..=0)?;
                let limit = self.config.max_str_len;
                let mut buf = Vec::new();
                let mut fh = f.lock().unwrap_or_else(|e| e.into_inner());
                (&mut fh.file)
                    .take(limit as u64 + 1)
                    .read_to_end(&mut buf)
                    .map_err(|e| err!(Io, "read failed: {e}"))?;
                if buf.len() > limit {
                    return Err(err!(Limit, "file is larger than {limit} bytes"));
                }
                if wants_bytes(hint) {
                    return Ok(Value::Bytes(buf.into()));
                }
                let s = String::from_utf8(buf)
                    .map_err(|e| err!(Value, "file is not valid UTF-8: {e}"))?;
                Ok(Value::Str(s.into()))
            }
            _ => Err(err!(Type, "'{}' has no method '{method}'", obj.type_of())),
        }
    }

    /// `place.method(args)` for methods that mutate their receiver, which
    /// must be a variable or an element of one.
    fn call_mutating_method(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
        env: &mut Env,
    ) -> Result<Value> {
        let Some((root, path)) = target_path(obj) else {
            return Err(err!(
                Type,
                "cannot call '{method}()' on a temporary value; assign it to a variable first"
            ));
        };
        let mut idxs = Vec::with_capacity(path.len());
        for e in &path {
            idxs.push(self.eval_expr(e, env, None)?);
        }
        let (scope_i, slot) = env
            .find(root)
            .map(|(i, s)| (i, s.clone()))
            .ok_or_else(|| err!(Name, "name '{root}' is not defined").at(obj.span))?;
        require_write(root, &slot, READ_WRITE_METHODS.contains(&method))
            .map_err(|e| e.at(obj.span))?;
        let place_ty = slot_type(
            &match &slot {
                Slot::Val(v, _) => v.type_of(),
                Slot::Cell(c, _) => c.ty.clone(),
            },
            &idxs,
        );
        let hint = match (&place_ty, method) {
            (Type::List(t), "append") | (Type::Set(t), "add" | "remove" | "discard") => {
                Some((**t).clone())
            }
            (Type::Dict(k, _), "pop") => Some((**k).clone()),
            (t @ Type::List(_), "extend") => Some(t.clone()),
            _ => None,
        };
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval_expr(a, env, hint.as_ref())?);
        }
        if method == "extend" && vals.len() == 1 {
            let items = self.collect(&vals[0])?;
            vals[0] = Value::new_list(Type::Unknown, items)?;
        }
        if let Slot::Cell(c, _) = &slot {
            check_no_cycle(c, root, &vals).map_err(|e| e.at(obj.span))?;
        }
        let apply = |v: &mut Value| with_place(v, &idxs, |c| mutate(c, method, vals));
        match slot {
            Slot::Val(..) => {
                let Some(Slot::Val(v, _)) = env.scopes[scope_i].get_mut(root) else {
                    unreachable!()
                };
                apply(v)
            }
            Slot::Cell(c, _) => apply(&mut c.guard()),
        }
    }

    // ------------------------------------------------------------- iteration

    fn next_item(&mut self, st: &mut IterState) -> Result<Option<Value>> {
        self.step()?;
        fn bump<T>(i: &mut usize, x: Option<T>) -> Option<T> {
            if x.is_some() {
                *i += 1;
            }
            x
        }
        Ok(match st {
            IterState::List(l, i) => bump(i, l.items.get(*i).cloned()),
            IterState::Tuple(t, i) => bump(i, t.get(*i).cloned()),
            IterState::Set(s, i) => bump(i, s.items.get_index(*i).map(Key::to_value)),
            IterState::Keys(d, i) => bump(i, d.map.get_index(*i).map(|(k, _)| k.to_value())),
            IterState::Values(d, i) => bump(i, d.map.get_index(*i).map(|(_, v)| v.clone())),
            IterState::Items(d, i) => match d.map.get_index(*i) {
                Some((k, v)) => {
                    *i += 1;
                    Some(Value::new_tuple(vec![k.to_value(), v.clone()])?)
                }
                None => None,
            },
            IterState::Str(s, i) => match s[*i..].chars().next() {
                Some(c) => {
                    *i += c.len_utf8();
                    Some(Value::Str(c.to_string().into()))
                }
                None => None,
            },
            IterState::Bytes(b, i) => bump(i, b.get(*i).map(|&x| Value::U8(x))),
            IterState::Range {
                cur,
                stop,
                step,
                ty,
            } => {
                if (*step > 0 && *cur < *stop) || (*step < 0 && *cur > *stop) {
                    let v = Value::int_of_type(ty, *cur).expect("within the range's bounds");
                    *cur += *step;
                    Some(v)
                } else {
                    None
                }
            }
            IterState::Map { f, inner } => match self.next_item(inner)? {
                Some(x) => {
                    let f = f.clone();
                    Some(self.call_value(&f, vec![x])?)
                }
                None => None,
            },
        })
    }

    fn collect(&mut self, v: &Value) -> Result<Vec<Value>> {
        let mut st = start_iter(v)?;
        let mut out = Vec::new();
        while let Some(x) = self.next_item(&mut st)? {
            out.push(x);
        }
        Ok(out)
    }

    fn iter_contains(&mut self, container: &Value, item: &Value) -> Result<bool> {
        let mut st = start_iter(container)?;
        while let Some(x) = self.next_item(&mut st)? {
            if ops::values_equal(&x, item)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// ------------------------------------------------------------------ helpers

fn lookup(env: &Env, name: &str) -> Result<Value> {
    match env.find(name) {
        Some((_, slot)) => read_slot(name, slot),
        None => Builtin::lookup(name)
            .map(Value::Builtin)
            .ok_or_else(|| err!(Name, "name '{name}' is not defined")),
    }
}

/// Copy (or, for locks, share) the outer variables a new function uses,
/// checking lock permissions against how the body uses them.
fn capture(env: &Env, free: &FreeVars, fname: &str) -> Result<Arc<Vec<(String, Slot)>>> {
    let mut out = Vec::new();
    for name in free.names() {
        if name == fname {
            continue; // a function always sees itself
        }
        match env.find(name) {
            None => {}
            Some((_, slot @ Slot::Val(..))) => out.push((name.clone(), slot.clone())),
            Some((_, Slot::Cell(c, access))) => {
                let sub = access.narrow(c.kind);
                let shown = format!("{}[{}]", c.kind.name(), c.ty);
                if free.writes.contains(name) && !sub.can_write() {
                    return Err(err!(
                        Permission,
                        "'{fname}' writes '{name}', but '{name}' is a {shown} and may only be read there"
                    ));
                }
                if free.reads.contains(name) && !sub.can_read() {
                    return Err(err!(
                        Permission,
                        "'{fname}' reads '{name}', but '{name}' is a {shown} and may only be written there"
                    ));
                }
                out.push((name.clone(), Slot::Cell(c.clone(), sub)));
            }
        }
    }
    Ok(Arc::new(out))
}

/// Whether the context wants `bytes` (for functions that can produce either
/// `str` or `bytes`).
fn wants_bytes(hint: Option<&Type>) -> bool {
    match hint {
        Some(Type::Bytes) => true,
        Some(Type::Optional(t)) => **t == Type::Bytes,
        _ => false,
    }
}

/// `module.name` for builtin modules.
fn module_attr(module: &Value, name: &str) -> Result<Value> {
    match (module, name) {
        (Value::Builtin(Builtin::Json), "loads") => Ok(Value::Builtin(Builtin::JsonLoads)),
        (Value::Builtin(Builtin::Json), "dumps") => Ok(Value::Builtin(Builtin::JsonDumps)),
        _ => Err(err!(Type, "module 'json' has no attribute '{name}'")),
    }
}

/// Whether values of type `t` may contain functions (and so, through their
/// captures, locks).
fn may_hold_fn(t: &Type) -> bool {
    match t {
        Type::Function(_) | Type::Any | Type::Unknown | Type::Iterable(_) => true,
        Type::List(e) | Type::Set(e) | Type::Optional(e) | Type::Lock(_, e) => may_hold_fn(e),
        Type::Dict(_, v) => may_hold_fn(v),
        Type::Tuple(ts) => ts.iter().any(may_hold_fn),
        _ => false,
    }
}

/// Refuse to store `vals` in `cell` if any of them can reach `cell` (through
/// functions' captured locks). Every other value is immutable or
/// copy-on-write, so a write into a lock is the only way a reference cycle
/// could form; checking here keeps the value graph acyclic, so everything
/// is freed by reference counting.
fn check_no_cycle(cell: &Arc<Cell>, name: &str, vals: &[Value]) -> Result<()> {
    let mut stack: Vec<Value> = vals
        .iter()
        .filter(|v| may_hold_fn(&v.type_of()))
        .cloned()
        .collect();
    let mut seen_fns: HashSet<*const Function> = HashSet::new();
    let mut seen_cells: HashSet<*const Cell> = HashSet::new();
    while let Some(v) = stack.pop() {
        match &v {
            Value::List(l) if may_hold_fn(&l.elem) => stack.extend(l.items.iter().cloned()),
            Value::Dict(d) if may_hold_fn(&d.val) => stack.extend(d.map.values().cloned()),
            Value::Tuple(items) => stack.extend(items.iter().cloned()),
            Value::Iter(view) => match &**view {
                IterView::Map { f, src, .. } => stack.extend([f.clone(), src.clone()]),
                IterView::Values(d) | IterView::Items(d) => stack.push(Value::Dict(d.clone())),
                IterView::Range { .. } | IterView::Keys(_) => {}
            },
            Value::Function(f) if seen_fns.insert(Arc::as_ptr(f)) => {
                for (_, slot) in f.captured.iter() {
                    match slot {
                        Slot::Val(v, _) => stack.push(v.clone()),
                        Slot::Cell(c, _) if Arc::ptr_eq(c, cell) => {
                            return Err(err!(
                                Value,
                                "storing function '{}' in '{name}' would create a reference \
                                 cycle: it captures '{name}'",
                                f.name
                            ));
                        }
                        Slot::Cell(c, _) if seen_cells.insert(Arc::as_ptr(c)) => {
                            stack.push(c.get())
                        }
                        Slot::Cell(..) => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn bind_targets(targets: &[String], item: Value) -> Result<Scope> {
    let mut scope = Scope::new();
    if let [name] = targets {
        scope.insert(name.clone(), Slot::val(item));
        return Ok(scope);
    }
    match &item {
        Value::Tuple(t) if t.len() == targets.len() => {
            for (name, v) in targets.iter().zip(t.iter()) {
                scope.insert(name.clone(), Slot::val(v.clone()));
            }
            Ok(scope)
        }
        other => Err(err!(
            Type,
            "cannot unpack '{}' into {} variables",
            other.type_of(),
            targets.len()
        )),
    }
}

fn start_iter(v: &Value) -> Result<IterState> {
    Ok(match v {
        Value::List(l) => IterState::List(l.clone(), 0),
        Value::Set(s) => IterState::Set(s.clone(), 0),
        Value::Dict(d) => IterState::Keys(d.clone(), 0),
        Value::Str(s) => IterState::Str(s.clone(), 0),
        Value::Bytes(b) => IterState::Bytes(b.clone(), 0),
        Value::Tuple(items) => {
            if v.type_of().iter_elem().is_none() {
                return Err(err!(
                    Type,
                    "cannot iterate over a mixed-type tuple '{}'",
                    v.type_of()
                ));
            }
            IterState::Tuple(items.clone(), 0)
        }
        Value::Iter(view) => match &**view {
            IterView::Range {
                start,
                stop,
                step,
                ty,
            } => IterState::Range {
                cur: *start,
                stop: *stop,
                step: *step,
                ty: ty.clone(),
            },
            IterView::Map { f, src, .. } => IterState::Map {
                f: f.clone(),
                inner: Box::new(start_iter(src)?),
            },
            IterView::Keys(d) => IterState::Keys(d.clone(), 0),
            IterView::Values(d) => IterState::Values(d.clone(), 0),
            IterView::Items(d) => IterState::Items(d.clone(), 0),
        },
        other => return Err(err!(Type, "'{}' is not iterable", other.type_of())),
    })
}

/// Type of `container[idxs[0]][idxs[1]]...`, for typing literals.
fn slot_type(container: &Type, idxs: &[Value]) -> Type {
    let mut t = container.clone();
    for _ in idxs {
        t = match t {
            Type::List(e) => *e,
            Type::Dict(_, v) => *v,
            _ => return Type::Unknown,
        };
    }
    t
}

fn param_hints(f: &Value) -> Vec<Option<Type>> {
    let known = |t: &Type| (*t != Type::Unknown).then(|| t.clone());
    match f {
        Value::Function(func) => func.params.iter().map(|(_, t)| t.clone()).collect(),
        Value::Builtin(Builtin::Convert(t)) => vec![Some(t.clone())],
        Value::Native(n) => n.params().unwrap_or_default().iter().map(known).collect(),
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
    let hint = match hint {
        Some(Type::Optional(t)) => Some(&**t),
        h => h,
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
        Some(Type::Optional(t)) if **t == Type::F32 => Value::F32(x as f32),
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

fn dict_key(d: &Dict, idx: &Value) -> Result<Key> {
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

/// Make `v` fit a collection slot of type `elem`, refining `elem` from
/// `Unknown` if needed.
fn fit_elem(elem: &mut Type, v: &mut Value, what: &str) -> Result<()> {
    if !elem.accepts(&v.type_of()) {
        return Err(err!(
            Type,
            "cannot store '{}' in '{what}[{elem}]'",
            v.type_of()
        ));
    }
    if !elem.has_unknown() {
        v.refine(elem);
    }
    if let Some(u) = elem.unify(&v.type_of()) {
        *elem = u;
    }
    Ok(())
}

/// Like [`fit_elem`] for dict values, which widen to `Any` unless the dict's
/// value type came from an annotation.
fn fit_dict_val(d: &mut Dict, v: &mut Value) -> Result<()> {
    if d.val.accepts(&v.type_of()) {
        if !d.val.has_unknown() {
            v.refine(&d.val);
        }
        if let Some(u) = d.val.unify(&v.type_of()) {
            d.val = u;
        }
        Ok(())
    } else if !d.val_fixed {
        d.val = Type::Any;
        Ok(())
    } else {
        Err(err!(
            Type,
            "cannot store '{}' in 'dict[{}, {}]'",
            v.type_of(),
            d.key,
            d.val
        ))
    }
}

/// Run `f` on `container[idxs[0]][idxs[1]]...`, copying shared collections
/// on write and keeping the containers' element types in sync.
fn with_place<R>(
    container: &mut Value,
    idxs: &[Value],
    f: impl FnOnce(&mut Value) -> Result<R>,
) -> Result<R> {
    let Some((idx, rest)) = idxs.split_first() else {
        return f(container);
    };
    match container {
        Value::List(l) => {
            let i = normalize_index(idx, l.items.len(), "list")?;
            let l = Arc::make_mut(l);
            let r = with_place(&mut l.items[i], rest, f)?;
            let t = l.items[i].type_of();
            l.elem = l
                .elem
                .unify(&t)
                .ok_or_else(|| err!(Type, "list element changed to incompatible type '{t}'"))?;
            Ok(r)
        }
        Value::Dict(d) => {
            let key = dict_key(d, idx)?;
            let d = Arc::make_mut(d);
            let child = d
                .map
                .get_mut(&key)
                .ok_or_else(|| err!(Key, "{}", idx.repr()))?;
            let r = with_place(child, rest, f)?;
            let t = child.type_of();
            d.val = d.val.unify(&t).unwrap_or(Type::Any);
            Ok(r)
        }
        other => Err(err!(
            Type,
            "'{}' does not support item assignment",
            other.type_of()
        )),
    }
}

/// `container[idx] = rhs` or `container[idx] op= rhs`.
fn set_item(
    container: &mut Value,
    idx: &Value,
    op: Option<BinOp>,
    rhs: Value,
    max_len: usize,
) -> Result<()> {
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
            let mut new = combine(Some(&l.items[i]), rhs)?;
            let mut elem = l.elem.clone();
            fit_elem(&mut elem, &mut new, "list")?;
            let l = Arc::make_mut(l);
            l.elem = elem;
            l.items[i] = new;
            Ok(())
        }
        Value::Dict(d) => {
            let key = dict_key(d, idx)?;
            let mut new = combine(d.map.get(&key), rhs)?;
            let mut copy = Dict {
                key: d.key.clone(),
                val: d.val.clone(),
                val_fixed: d.val_fixed,
                map: Default::default(),
            };
            fit_dict_val(&mut copy, &mut new)?;
            let d = Arc::make_mut(d);
            d.val = copy.val;
            d.key = d.key.unify(&idx.type_of()).expect("checked by dict_key");
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

/// Apply a mutating method to its receiver.
fn mutate(target: &mut Value, method: &str, args: Vec<Value>) -> Result<Value> {
    let n = args.len();
    let arity = |range: std::ops::RangeInclusive<usize>| {
        if range.contains(&n) {
            Ok(())
        } else {
            Err(err!(Type, "{method}() got {n} argument(s)"))
        }
    };
    let ty = target.type_of();
    let mut args = args.into_iter();
    match (target, method) {
        (Value::List(l), "append") => {
            arity(1..=1)?;
            let mut v = args.next().unwrap();
            let mut elem = l.elem.clone();
            fit_elem(&mut elem, &mut v, "list")?;
            check_depth(&elem)?;
            let l = Arc::make_mut(l);
            l.elem = elem;
            l.items.push(v);
            Ok(Value::None)
        }
        (Value::List(l), "extend") => {
            arity(1..=1)?;
            let Value::List(src) = args.next().unwrap() else {
                unreachable!("collected by the caller")
            };
            let mut elem = l.elem.clone();
            let mut items = src.items.clone();
            for v in &mut items {
                fit_elem(&mut elem, v, "list")?;
            }
            check_depth(&elem)?;
            let l = Arc::make_mut(l);
            l.elem = elem;
            l.items.extend(items);
            Ok(Value::None)
        }
        (Value::List(l), "pop") => {
            arity(0..=1)?;
            let i = match args.next() {
                Some(idx) => normalize_index(&idx, l.items.len(), "pop")?,
                None if l.items.is_empty() => return Err(err!(Index, "pop from empty list")),
                None => l.items.len() - 1,
            };
            Ok(Arc::make_mut(l).items.remove(i))
        }
        (Value::Set(s), "add" | "remove" | "discard") => {
            arity(1..=1)?;
            let mut v = args.next().unwrap();
            let mut elem = s.elem.clone();
            fit_elem(&mut elem, &mut v, "set")?;
            if !elem.is_hashable() {
                return Err(err!(Type, "unhashable type: '{elem}'"));
            }
            let key = Key::from_value(&v)?;
            let s = Arc::make_mut(s);
            match method {
                "add" => {
                    s.elem = elem;
                    s.items.insert(key);
                }
                "remove" => {
                    if !s.items.shift_remove(&key) {
                        return Err(err!(Key, "{}", v.repr()));
                    }
                }
                _ => {
                    s.items.shift_remove(&key);
                }
            }
            Ok(Value::None)
        }
        (Value::Dict(d), "pop") => {
            arity(1..=2)?;
            let k = args.next().unwrap();
            let key = dict_key(d, &k)?;
            match Arc::make_mut(d).map.shift_remove(&key) {
                Some(v) => Ok(v),
                None => args.next().ok_or_else(|| err!(Key, "{}", k.repr())),
            }
        }
        (Value::List(l), "clear") => {
            arity(0..=0)?;
            Arc::make_mut(l).items.clear();
            Ok(Value::None)
        }
        (Value::Set(s), "clear") => {
            arity(0..=0)?;
            Arc::make_mut(s).items.clear();
            Ok(Value::None)
        }
        (Value::Dict(d), "clear") => {
            arity(0..=0)?;
            Arc::make_mut(d).map.clear();
            Ok(Value::None)
        }
        _ => Err(err!(Type, "'{ty}' has no method '{method}'")),
    }
}

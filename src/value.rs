//! Runtime values.
//!
//! Collections have *value semantics* (like Rust's owned `Vec`/`HashMap`):
//! they are `Arc`-shared and copied on write via [`Arc::make_mut`], so
//! `b = a; b[0] = 1` never changes `a`. Everything is `Send + Sync`, and
//! value nesting depth is bounded by [`MAX_TYPE_DEPTH`].

use std::fmt::{self, Write as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use indexmap::{IndexMap, IndexSet};

use crate::error::{Result, err};
use crate::interp::{Builtin, Function};
use crate::native::NativeFn;
use crate::types::{MAX_TYPE_DEPTH, Type};

#[derive(Clone)]
pub enum Value {
    None,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
    List(Arc<List>),
    Dict(Arc<Dict>),
    Set(Arc<Set>),
    Tuple(Arc<[Value]>),
    /// A lazy, re-iterable view.
    Iter(Arc<IterView>),
    Function(Arc<Function>),
    Builtin(Builtin),
    Native(Arc<NativeFn>),
    Path(Arc<PathBuf>),
    File(Arc<Mutex<FileHandle>>),
}

#[derive(Clone, Debug)]
pub struct List {
    pub elem: Type,
    pub items: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct Dict {
    pub key: Type,
    pub val: Type,
    /// Whether `val` came from an annotation. Inferred dicts widen their
    /// value type to `Any` when given a value of a new type (JSON-style
    /// objects); annotated ones reject it.
    pub val_fixed: bool,
    pub map: IndexMap<Key, Value>,
}

#[derive(Clone, Debug)]
pub struct Set {
    pub elem: Type,
    pub items: IndexSet<Key>,
}

/// Lazy iterables. Each holds a snapshot of its source (value semantics),
/// so iterating it never observes later mutation and can be repeated.
#[derive(Debug)]
pub enum IterView {
    Range {
        start: i128,
        stop: i128,
        step: i128,
        ty: Type,
    },
    Map {
        f: Value,
        src: Value,
        elem: Type,
    },
    Keys(Arc<Dict>),
    Values(Arc<Dict>),
    Items(Arc<Dict>),
}

impl IterView {
    pub fn elem_type(&self) -> Type {
        match self {
            IterView::Range { ty, .. } => ty.clone(),
            IterView::Map { elem, .. } => elem.clone(),
            IterView::Keys(d) => d.key.clone(),
            IterView::Values(d) => d.val.clone(),
            IterView::Items(d) => Type::Tuple(vec![d.key.clone(), d.val.clone()]),
        }
    }

    /// Number of items, if known without running user code.
    pub fn known_len(&self) -> Option<usize> {
        match self {
            IterView::Range {
                start, stop, step, ..
            } => Some(range_len(*start, *stop, *step)),
            IterView::Map { .. } => None,
            IterView::Keys(d) | IterView::Values(d) | IterView::Items(d) => Some(d.map.len()),
        }
    }
}

pub fn range_len(start: i128, stop: i128, step: i128) -> usize {
    let n = if step > 0 {
        (stop - start + step - 1).div_euclid(step)
    } else {
        (start - stop - step - 1).div_euclid(-step)
    };
    n.max(0) as usize
}

/// An open file, or whatever the host's [`Opener`](crate::Opener) returned
/// for that path.
pub struct FileHandle {
    pub path: PathBuf,
    pub reader: crate::fs::Reader,
}

impl fmt::Debug for FileHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<file {}>", self.path.display())
    }
}

/// The hashable subset of values, usable as dict keys and set elements.
/// Floats are deliberately excluded (as in Rust), so `Eq`/`Hash` are sound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    None,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
    Path(Arc<PathBuf>),
    Tuple(Arc<[Key]>),
}

impl Key {
    pub fn from_value(v: &Value) -> Result<Key> {
        Ok(match v {
            Value::None => Key::None,
            Value::Bool(b) => Key::Bool(*b),
            Value::I8(x) => Key::I8(*x),
            Value::I16(x) => Key::I16(*x),
            Value::I32(x) => Key::I32(*x),
            Value::I64(x) => Key::I64(*x),
            Value::U8(x) => Key::U8(*x),
            Value::U16(x) => Key::U16(*x),
            Value::U32(x) => Key::U32(*x),
            Value::U64(x) => Key::U64(*x),
            Value::Str(s) => Key::Str(s.clone()),
            Value::Bytes(b) => Key::Bytes(b.clone()),
            Value::Path(p) => Key::Path(p.clone()),
            Value::Tuple(items) => Key::Tuple(
                items
                    .iter()
                    .map(Key::from_value)
                    .collect::<Result<Vec<_>>>()?
                    .into(),
            ),
            other => return Err(err!(Type, "unhashable type: '{}'", other.type_of())),
        })
    }

    pub fn to_value(&self) -> Value {
        match self {
            Key::None => Value::None,
            Key::Bool(b) => Value::Bool(*b),
            Key::I8(x) => Value::I8(*x),
            Key::I16(x) => Value::I16(*x),
            Key::I32(x) => Value::I32(*x),
            Key::I64(x) => Value::I64(*x),
            Key::U8(x) => Value::U8(*x),
            Key::U16(x) => Value::U16(*x),
            Key::U32(x) => Value::U32(*x),
            Key::U64(x) => Value::U64(*x),
            Key::Str(s) => Value::Str(s.clone()),
            Key::Bytes(b) => Value::Bytes(b.clone()),
            Key::Path(p) => Value::Path(p.clone()),
            Key::Tuple(items) => Value::Tuple(items.iter().map(Key::to_value).collect()),
        }
    }
}

pub(crate) fn check_depth(ty: &Type) -> Result<()> {
    if ty.depth() >= MAX_TYPE_DEPTH {
        Err(crate::Error::limit(crate::LimitExceeded::NestingDepth(
            MAX_TYPE_DEPTH,
        )))
    } else {
        Ok(())
    }
}

/// Unify the types of `items`, starting from `elem`. Errors on mixed types.
fn unify_all(what: &str, mut elem: Type, types: impl Iterator<Item = Type>) -> Result<Type> {
    for t in types {
        elem = elem
            .unify(&t)
            .ok_or_else(|| err!(Type, "{what} cannot mix types '{elem}' and '{t}'"))?;
    }
    Ok(elem)
}

impl Value {
    /// Build a list, checking that every item has a compatible type.
    pub fn new_list(elem: Type, items: Vec<Value>) -> Result<Value> {
        let elem = unify_all("list", elem, items.iter().map(Value::type_of))?;
        check_depth(&elem)?;
        Ok(Value::List(Arc::new(List { elem, items })))
    }

    pub fn new_set(elem: Type, items: Vec<Value>) -> Result<Value> {
        let elem = unify_all("set", elem, items.iter().map(Value::type_of))?;
        check_depth(&elem)?;
        if !elem.is_hashable() {
            return Err(err!(Type, "unhashable type: '{elem}'"));
        }
        let items = items.iter().map(Key::from_value).collect::<Result<_>>()?;
        Ok(Value::Set(Arc::new(Set { elem, items })))
    }

    /// Build a dict. Keys must share a type; values widen to `Any` if mixed.
    pub fn new_dict(key: Type, val: Type, entries: Vec<(Value, Value)>) -> Result<Value> {
        let key = unify_all("dict keys", key, entries.iter().map(|(k, _)| k.type_of()))?;
        let val = entries.iter().fold(val, |acc, (_, v)| {
            acc.unify(&v.type_of()).unwrap_or(Type::Any)
        });
        check_depth(&key)?;
        check_depth(&val)?;
        if !key.is_hashable() {
            return Err(err!(Type, "unhashable type: '{key}'"));
        }
        let mut map = IndexMap::with_capacity(entries.len());
        for (k, v) in entries {
            map.insert(Key::from_value(&k)?, v);
        }
        Ok(Value::Dict(Arc::new(Dict {
            key,
            val,
            val_fixed: false,
            map,
        })))
    }

    pub fn new_tuple(items: Vec<Value>) -> Result<Value> {
        let v = Value::Tuple(items.into());
        check_depth(&v.type_of())?;
        Ok(v)
    }

    pub fn str(s: impl Into<Arc<str>>) -> Value {
        Value::Str(s.into())
    }

    pub fn type_of(&self) -> Type {
        match self {
            Value::None => Type::None,
            Value::Bool(_) => Type::Bool,
            Value::I8(_) => Type::I8,
            Value::I16(_) => Type::I16,
            Value::I32(_) => Type::I32,
            Value::I64(_) => Type::I64,
            Value::U8(_) => Type::U8,
            Value::U16(_) => Type::U16,
            Value::U32(_) => Type::U32,
            Value::U64(_) => Type::U64,
            Value::F32(_) => Type::F32,
            Value::F64(_) => Type::F64,
            Value::Str(_) => Type::Str,
            Value::Bytes(_) => Type::Bytes,
            Value::List(l) => Type::List(Box::new(l.elem.clone())),
            Value::Dict(d) => Type::Dict(Box::new(d.key.clone()), Box::new(d.val.clone())),
            Value::Set(s) => Type::Set(Box::new(s.elem.clone())),
            Value::Tuple(items) => Type::Tuple(items.iter().map(Value::type_of).collect()),
            Value::Iter(v) => Type::Iterable(Box::new(v.elem_type())),
            Value::Function(f) => f.signature(),
            Value::Builtin(_) => Type::Function(None),
            Value::Native(n) => n.signature(),
            Value::Path(_) => Type::Path,
            Value::File(_) => Type::File,
        }
    }

    pub fn is_callable(&self) -> bool {
        matches!(
            self,
            Value::Function(_) | Value::Builtin(_) | Value::Native(_)
        )
    }

    /// Adopt the annotated type `ty` (which must accept this value): fills
    /// in `Unknown` element types from empty collections and fixes dict
    /// value types so they stop widening.
    pub fn refine(&mut self, ty: &Type) {
        let ty = match ty {
            Type::Optional(t) => &**t,
            t => t,
        };
        if ty.has_unknown() {
            return;
        }
        match (self, ty) {
            (Value::List(l), Type::List(t)) if l.elem != **t => {
                let l = Arc::make_mut(l);
                l.elem = (**t).clone();
                l.items.iter_mut().for_each(|v| v.refine(t));
            }
            (Value::Set(s), Type::Set(t)) if s.elem != **t => {
                Arc::make_mut(s).elem = (**t).clone();
            }
            (Value::Dict(d), Type::Dict(k, v)) if d.key != **k || d.val != **v || !d.val_fixed => {
                let d = Arc::make_mut(d);
                d.key = (**k).clone();
                d.val = (**v).clone();
                d.val_fixed = true;
                d.map.values_mut().for_each(|x| x.refine(v));
            }
            (Value::Tuple(items), Type::Tuple(ts)) if items.len() == ts.len() => {
                let mut new: Vec<Value> = items.to_vec();
                new.iter_mut().zip(ts).for_each(|(v, t)| v.refine(t));
                *items = new.into();
            }
            _ => {}
        }
    }

    pub fn as_i128(&self) -> Option<i128> {
        Some(match *self {
            Value::I8(x) => x.into(),
            Value::I16(x) => x.into(),
            Value::I32(x) => x.into(),
            Value::I64(x) => x.into(),
            Value::U8(x) => x.into(),
            Value::U16(x) => x.into(),
            Value::U32(x) => x.into(),
            Value::U64(x) => x.into(),
            _ => return None,
        })
    }

    /// Build an integer of type `ty` from `v`, if it fits.
    pub fn int_of_type(ty: &Type, v: i128) -> Option<Value> {
        Some(match ty {
            Type::I8 => Value::I8(v.try_into().ok()?),
            Type::I16 => Value::I16(v.try_into().ok()?),
            Type::I32 => Value::I32(v.try_into().ok()?),
            Type::I64 => Value::I64(v.try_into().ok()?),
            Type::U8 => Value::U8(v.try_into().ok()?),
            Type::U16 => Value::U16(v.try_into().ok()?),
            Type::U32 => Value::U32(v.try_into().ok()?),
            Type::U64 => Value::U64(v.try_into().ok()?),
            _ => return None,
        })
    }

    /// Text used by `print` and `str()`: strings are written raw, everything
    /// else as its repr.
    pub fn display(&self) -> String {
        match self {
            Value::Str(s) => s.to_string(),
            other => other.repr(),
        }
    }

    pub fn repr(&self) -> String {
        let mut out = String::new();
        self.write_repr(&mut out);
        out
    }

    fn write_repr(&self, out: &mut String) {
        fn seq<'a>(out: &mut String, items: impl Iterator<Item = &'a Value>) {
            for (i, v) in items.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                v.write_repr(out);
            }
        }
        fn keys<'a>(out: &mut String, items: impl Iterator<Item = &'a Key>) {
            for (i, k) in items.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                k.to_value().write_repr(out);
            }
        }
        match self {
            Value::None => out.push_str("None"),
            Value::Bool(true) => out.push_str("True"),
            Value::Bool(false) => out.push_str("False"),
            Value::I8(x) => write!(out, "{x}").unwrap(),
            Value::I16(x) => write!(out, "{x}").unwrap(),
            Value::I32(x) => write!(out, "{x}").unwrap(),
            Value::I64(x) => write!(out, "{x}").unwrap(),
            Value::U8(x) => write!(out, "{x}").unwrap(),
            Value::U16(x) => write!(out, "{x}").unwrap(),
            Value::U32(x) => write!(out, "{x}").unwrap(),
            Value::U64(x) => write!(out, "{x}").unwrap(),
            Value::F32(x) => write_float(out, f64::from(*x), format!("{x:?}")),
            Value::F64(x) => write_float(out, *x, format!("{x:?}")),
            Value::Str(s) => write_str_repr(out, s),
            Value::Bytes(b) => write_bytes_repr(out, b),
            Value::List(l) => {
                out.push('[');
                seq(out, l.items.iter());
                out.push(']');
            }
            Value::Tuple(items) => {
                out.push('(');
                seq(out, items.iter());
                if items.len() == 1 {
                    out.push(',');
                }
                out.push(')');
            }
            Value::Set(s) if s.items.is_empty() => out.push_str("set()"),
            Value::Set(s) => {
                out.push('{');
                keys(out, s.items.iter());
                out.push('}');
            }
            Value::Dict(d) => {
                out.push('{');
                for (i, (k, v)) in d.map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    k.to_value().write_repr(out);
                    out.push_str(": ");
                    v.write_repr(out);
                }
                out.push('}');
            }
            Value::Iter(view) => match &**view {
                IterView::Range {
                    start, stop, step, ..
                } => {
                    if *step == 1 {
                        write!(out, "range({start}, {stop})").unwrap()
                    } else {
                        write!(out, "range({start}, {stop}, {step})").unwrap()
                    }
                }
                IterView::Map { .. } => out.push_str("<map object>"),
                IterView::Keys(d) => {
                    out.push_str("dict_keys([");
                    keys(out, d.map.keys());
                    out.push_str("])");
                }
                IterView::Values(d) => {
                    out.push_str("dict_values([");
                    seq(out, d.map.values());
                    out.push_str("])");
                }
                IterView::Items(d) => {
                    out.push_str("dict_items([");
                    for (i, (k, v)) in d.map.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push('(');
                        k.to_value().write_repr(out);
                        out.push_str(", ");
                        v.write_repr(out);
                        out.push(')');
                    }
                    out.push_str("])");
                }
            },
            Value::Function(f) => write!(out, "<function {}>", f.name).unwrap(),
            Value::Builtin(Builtin::Json) => out.push_str("<module 'json'>"),
            Value::Builtin(b) => write!(out, "<built-in function {}>", b.name()).unwrap(),
            Value::Native(n) => write!(out, "<native function {}>", n.name).unwrap(),
            Value::Path(p) => {
                out.push_str("Path(");
                write_str_repr(out, &p.to_string_lossy());
                out.push(')');
            }
            Value::File(f) => {
                let f = f.lock().unwrap_or_else(|e| e.into_inner());
                out.push_str("<file ");
                write_str_repr(out, &f.path.to_string_lossy());
                out.push('>');
            }
        }
    }
}

fn write_float(out: &mut String, x: f64, debug: String) {
    if x.is_nan() {
        out.push_str("nan");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "inf" } else { "-inf" });
    } else {
        out.push_str(&debug);
    }
}

fn write_str_repr(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => write!(out, "\\x{:02x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push(quote);
}

fn write_bytes_repr(out: &mut String, b: &[u8]) {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    out.push('b');
    out.push(quote as char);
    for &c in b {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c as char);
            }
            0x20..=0x7e => out.push(c as char),
            c => write!(out, "\\x{c:02x}").unwrap(),
        }
    }
    out.push(quote as char);
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.repr())
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

/// Structural equality (used by tests and hosts). Values of different types
/// are never equal; scripts get a `TypeError` instead (see `ops::compare`).
impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        crate::ops::values_equal(self, other).unwrap_or(false)
    }
}

// ---- conversions for embedders

macro_rules! from_prim {
    ($($t:ty => $v:ident),*) => {$(
        impl From<$t> for Value {
            fn from(x: $t) -> Value { Value::$v(x) }
        }
        impl TryFrom<&Value> for $t {
            type Error = crate::Error;
            fn try_from(v: &Value) -> Result<$t> {
                match v {
                    Value::$v(x) => Ok(*x),
                    other => Err(err!(Type, "expected {}, got '{}'", stringify!($t), other.type_of())),
                }
            }
        }
    )*};
}
from_prim!(bool => Bool, i8 => I8, i16 => I16, i32 => I32, i64 => I64,
           u8 => U8, u16 => U16, u32 => U32, u64 => U64, f32 => F32, f64 => F64);

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.into())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s.into())
    }
}

impl From<&[u8]> for Value {
    fn from(b: &[u8]) -> Value {
        Value::Bytes(b.into())
    }
}

impl From<()> for Value {
    fn from(_: ()) -> Value {
        Value::None
    }
}

impl From<PathBuf> for Value {
    fn from(p: PathBuf) -> Value {
        Value::Path(Arc::new(p))
    }
}

impl TryFrom<&Value> for String {
    type Error = crate::Error;
    fn try_from(v: &Value) -> Result<String> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            other => Err(err!(Type, "expected str, got '{}'", other.type_of())),
        }
    }
}

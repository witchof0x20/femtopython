// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

//! Runtime types and type annotations.

use std::fmt;

/// Maximum nesting depth of a type such as `list[list[...]]`. Values are
/// bounded by the same limit, so every recursive walk over a value (repr,
/// equality, drop) has bounded stack usage.
pub const MAX_TYPE_DEPTH: usize = 32;

/// How a lock-annotated variable is shared with nested functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockKind {
    /// `rwlock[T]`: nested functions may read and write.
    ReadWrite,
    /// `rlock[T]`: nested functions may only read.
    Read,
    /// `wlock[T]`: nested functions may only write.
    Write,
}

impl LockKind {
    pub fn name(self) -> &'static str {
        match self {
            LockKind::ReadWrite => "rwlock",
            LockKind::Read => "rlock",
            LockKind::Write => "wlock",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnSig {
    pub params: Vec<Type>,
    pub ret: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Element type of an empty collection, or a parameter/return type that
    /// was never annotated. Never written by users; compatible with
    /// everything and refined by the first concrete type it meets.
    Unknown,
    /// Explicitly dynamic. Accepts any value.
    Any,
    None,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Str,
    Bytes,
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Set(Box<Type>),
    Tuple(Vec<Type>),
    /// `Optional[T]`: either `None` or a `T`.
    Optional(Box<Type>),
    /// A lazy, re-iterable view (`map`, `range`, `dict.keys()`, ...).
    Iterable(Box<Type>),
    /// `Callable` (any function) or `Callable[[A, B], R]`.
    Function(Option<Box<FnSig>>),
    Path,
    File,
    /// `rwlock[T]` / `rlock[T]` / `wlock[T]`. Only valid as the outermost
    /// annotation of a variable declaration; never the type of a value.
    Lock(LockKind, Box<Type>),
}

/// An argument inside `name[...]` in an annotation. `Many` is the
/// bracketed parameter list in `Callable[[A, B], R]`.
#[derive(Debug)]
pub enum TypeArg {
    One(Type),
    Many(Vec<Type>),
}

impl Type {
    /// Resolve an annotation like `u8`, `list[int]`, `Optional[str]` or
    /// `Callable[[int], bool]`.
    pub fn from_name(name: &str, args: Vec<TypeArg>) -> Result<Type, String> {
        let n = args.len();
        let mut single = Vec::with_capacity(n);
        let mut lists = Vec::new();
        for a in args {
            match a {
                TypeArg::One(Type::Lock(k, _)) => {
                    return Err(format!(
                        "'{}' may only annotate a variable declaration",
                        k.name()
                    ));
                }
                TypeArg::One(t) => single.push(t),
                TypeArg::Many(ts) => lists.push(ts),
            }
        }
        if !lists.is_empty() && name != "Callable" {
            return Err(format!("type '{name}' does not take a parameter list"));
        }
        let arity = |want: usize| {
            if n == want {
                Ok(())
            } else {
                Err(format!(
                    "type '{name}' takes {want} type argument(s), got {n}"
                ))
            }
        };
        let mut it = single.into_iter();
        let mut one = || it.next().expect("arity checked");
        let ty = match name {
            "bool" => arity(0).map(|_| Type::Bool)?,
            "i8" => arity(0).map(|_| Type::I8)?,
            "i16" => arity(0).map(|_| Type::I16)?,
            "i32" => arity(0).map(|_| Type::I32)?,
            "i64" | "int" => arity(0).map(|_| Type::I64)?,
            "u8" => arity(0).map(|_| Type::U8)?,
            "u16" => arity(0).map(|_| Type::U16)?,
            "u32" => arity(0).map(|_| Type::U32)?,
            "u64" => arity(0).map(|_| Type::U64)?,
            "f32" => arity(0).map(|_| Type::F32)?,
            "f64" | "float" => arity(0).map(|_| Type::F64)?,
            "str" => arity(0).map(|_| Type::Str)?,
            "bytes" => arity(0).map(|_| Type::Bytes)?,
            "Path" => arity(0).map(|_| Type::Path)?,
            "Any" => arity(0).map(|_| Type::Any)?,
            "list" => {
                arity(1)?;
                Type::List(Box::new(one()))
            }
            "Iterable" => {
                arity(1)?;
                Type::Iterable(Box::new(one()))
            }
            "Optional" => {
                arity(1)?;
                Type::optional(one())
            }
            "set" => {
                arity(1)?;
                let elem = one();
                if !elem.is_hashable() {
                    return Err(format!("set element type '{elem}' is not hashable"));
                }
                Type::Set(Box::new(elem))
            }
            "dict" => {
                arity(2)?;
                let (k, v) = (one(), one());
                if !k.is_hashable() {
                    return Err(format!("dict key type '{k}' is not hashable"));
                }
                Type::Dict(Box::new(k), Box::new(v))
            }
            "tuple" => Type::Tuple(it.collect()),
            "Callable" => match (n, lists.pop()) {
                (0, None) => Type::Function(None),
                (2, Some(params)) => Type::Function(Some(Box::new(FnSig { params, ret: one() }))),
                _ => return Err("expected Callable or Callable[[A, ...], R]".into()),
            },
            "rwlock" | "rlock" | "wlock" => {
                arity(1)?;
                let kind = match name {
                    "rwlock" => LockKind::ReadWrite,
                    "rlock" => LockKind::Read,
                    _ => LockKind::Write,
                };
                Type::Lock(kind, Box::new(one()))
            }
            _ => return Err(format!("unknown type '{name}'")),
        };
        ty.check_depth().map_err(|e| e.to_string())?;
        Ok(ty)
    }

    /// `Optional[T]`, flattening `Optional[Optional[T]]` and `Optional[None]`.
    pub fn optional(t: Type) -> Type {
        match t {
            Type::None | Type::Optional(_) | Type::Any => t,
            t => Type::Optional(Box::new(t)),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Type::List(t)
            | Type::Set(t)
            | Type::Optional(t)
            | Type::Iterable(t)
            | Type::Lock(_, t) => 1 + t.depth(),
            Type::Dict(k, v) => 1 + k.depth().max(v.depth()),
            Type::Tuple(ts) => 1 + ts.iter().map(Type::depth).max().unwrap_or(0),
            Type::Function(Some(sig)) => {
                1 + sig
                    .params
                    .iter()
                    .chain([&sig.ret])
                    .map(Type::depth)
                    .max()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    pub fn check_depth(&self) -> crate::Result<()> {
        if self.depth() > MAX_TYPE_DEPTH {
            Err(crate::Error::limit(crate::LimitExceeded::NestingDepth(
                MAX_TYPE_DEPTH,
            )))
        } else {
            Ok(())
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    pub fn has_unknown(&self) -> bool {
        match self {
            Type::Unknown => true,
            Type::List(t)
            | Type::Set(t)
            | Type::Optional(t)
            | Type::Iterable(t)
            | Type::Lock(_, t) => t.has_unknown(),
            Type::Dict(k, v) => k.has_unknown() || v.has_unknown(),
            Type::Tuple(ts) => ts.iter().any(Type::has_unknown),
            Type::Function(Some(sig)) => {
                sig.params.iter().any(Type::has_unknown) || sig.ret.has_unknown()
            }
            _ => false,
        }
    }

    /// Whether values of this type may be dict keys / set elements.
    /// Floats are excluded, as in Rust (`f64: !Hash + !Eq`).
    pub fn is_hashable(&self) -> bool {
        match self {
            Type::Unknown | Type::None | Type::Bool | Type::Str | Type::Bytes | Type::Path => true,
            t if t.is_int() => true,
            Type::Tuple(ts) => ts.iter().all(Type::is_hashable),
            Type::Optional(t) => t.is_hashable(),
            _ => false,
        }
    }

    /// Can a value of type `actual` be used where `self` is expected?
    pub fn accepts(&self, actual: &Type) -> bool {
        match (self, actual) {
            (Type::Any | Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Optional(_), Type::None) => true,
            (Type::Optional(e), Type::Optional(a)) => e.accepts(a),
            (Type::Optional(e), a) => e.accepts(a),
            // Collections have value semantics, so covariance is sound.
            (Type::List(e), Type::List(a))
            | (Type::Set(e), Type::Set(a))
            | (Type::Iterable(e), Type::Iterable(a)) => e.accepts(a),
            (Type::Dict(ek, ev), Type::Dict(ak, av)) => ek.accepts(ak) && ev.accepts(av),
            (Type::Tuple(e), Type::Tuple(a)) => {
                e.len() == a.len() && e.iter().zip(a).all(|(e, a)| e.accepts(a))
            }
            (Type::Function(None), Type::Function(_))
            | (Type::Function(Some(_)), Type::Function(None)) => true,
            (Type::Function(Some(e)), Type::Function(Some(a))) => {
                e.params.len() == a.params.len()
                    // Parameters are contravariant: the function must accept
                    // everything the expected signature may pass it.
                    && e.params.iter().zip(&a.params).all(|(e, a)| a.accepts(e))
                    && e.ret.accepts(&a.ret)
            }
            (e, a) => e == a,
        }
    }

    /// The most specific type describing values of both types, used to infer
    /// collection element types. `None` if they conflict.
    pub fn unify(&self, other: &Type) -> Option<Type> {
        Some(match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => t.clone(),
            (Type::Any, _) | (_, Type::Any) => Type::Any,
            (Type::None, Type::None) => Type::None,
            (Type::None, t) | (t, Type::None) => Type::optional(t.clone()),
            (Type::Optional(a), Type::Optional(b)) => Type::optional(a.unify(b)?),
            (Type::Optional(a), b) | (b, Type::Optional(a)) => Type::optional(a.unify(b)?),
            (Type::List(a), Type::List(b)) => Type::List(Box::new(a.unify(b)?)),
            (Type::Set(a), Type::Set(b)) => Type::Set(Box::new(a.unify(b)?)),
            (Type::Iterable(a), Type::Iterable(b)) => Type::Iterable(Box::new(a.unify(b)?)),
            (Type::Dict(ka, va), Type::Dict(kb, vb)) => Type::Dict(
                Box::new(ka.unify(kb)?),
                // Dict values are heterogeneous (think JSON objects).
                Box::new(va.unify(vb).unwrap_or(Type::Any)),
            ),
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => Type::Tuple(
                a.iter()
                    .zip(b)
                    .map(|(x, y)| x.unify(y))
                    .collect::<Option<_>>()?,
            ),
            (Type::Function(a), Type::Function(b)) => {
                Type::Function(if a == b { a.clone() } else { None })
            }
            (a, b) if a == b => a.clone(),
            _ => return None,
        })
    }

    /// Element type yielded when iterating a value of this type, if it is
    /// iterable.
    pub fn iter_elem(&self) -> Option<Type> {
        match self {
            Type::List(t) | Type::Set(t) | Type::Iterable(t) => Some((**t).clone()),
            Type::Dict(k, _) => Some((**k).clone()),
            Type::Str => Some(Type::Str),
            Type::Bytes => Some(Type::U8),
            Type::Tuple(ts) => {
                let mut it = ts.iter();
                let first = it.next().cloned().unwrap_or(Type::Unknown);
                it.try_fold(first, |acc, t| {
                    (acc == *t || *t == Type::Unknown).then_some(acc)
                })
            }
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn list(f: &mut fmt::Formatter<'_>, ts: &[Type]) -> fmt::Result {
            for (i, t) in ts.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{t}")?;
            }
            Ok(())
        }
        match self {
            Type::Unknown => write!(f, "?"),
            Type::Any => write!(f, "Any"),
            Type::None => write!(f, "None"),
            Type::Bool => write!(f, "bool"),
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "int"),
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "float"),
            Type::Str => write!(f, "str"),
            Type::Bytes => write!(f, "bytes"),
            Type::List(t) => write!(f, "list[{t}]"),
            Type::Set(t) => write!(f, "set[{t}]"),
            Type::Dict(k, v) => write!(f, "dict[{k}, {v}]"),
            Type::Optional(t) => write!(f, "Optional[{t}]"),
            Type::Iterable(t) => write!(f, "Iterable[{t}]"),
            Type::Tuple(ts) => {
                write!(f, "tuple[")?;
                list(f, ts)?;
                write!(f, "]")
            }
            Type::Function(None) => write!(f, "Callable"),
            Type::Function(Some(sig)) => {
                write!(f, "Callable[[")?;
                list(f, &sig.params)?;
                write!(f, "], {}]", sig.ret)
            }
            Type::Path => write!(f, "Path"),
            Type::File => write!(f, "File"),
            Type::Lock(k, t) => write!(f, "{}[{t}]", k.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(t: Type) -> Type {
        Type::optional(t)
    }

    #[test]
    fn optional_unification() {
        assert_eq!(Type::I64.unify(&Type::None), Some(opt(Type::I64)));
        assert_eq!(opt(Type::I64).unify(&Type::I64), Some(opt(Type::I64)));
        assert_eq!(Type::I64.unify(&Type::Str), None);
        assert!(opt(Type::U8).accepts(&Type::None));
        assert!(opt(Type::U8).accepts(&Type::U8));
        assert!(!Type::U8.accepts(&Type::None));
    }

    #[test]
    fn dict_values_widen_to_any() {
        let a = Type::Dict(Box::new(Type::Str), Box::new(Type::I64));
        let b = Type::Dict(Box::new(Type::Str), Box::new(Type::Str));
        assert_eq!(
            a.unify(&b),
            Some(Type::Dict(Box::new(Type::Str), Box::new(Type::Any)))
        );
    }

    #[test]
    fn callable_variance() {
        let int_to_int = Type::Function(Some(Box::new(FnSig {
            params: vec![Type::I64],
            ret: Type::I64,
        })));
        let any_to_int = Type::Function(Some(Box::new(FnSig {
            params: vec![Type::Any],
            ret: Type::I64,
        })));
        assert!(int_to_int.accepts(&any_to_int));
        assert!(!any_to_int.accepts(&int_to_int));
        assert!(Type::Function(None).accepts(&int_to_int));
    }
}

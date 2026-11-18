//! Runtime types and type annotations.

use std::fmt;

/// Maximum nesting depth of a type such as `list[list[...]]`. Values are
/// bounded by the same limit, so every recursive walk over a value (repr,
/// equality, drop) has bounded stack usage.
pub const MAX_TYPE_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Element type of an empty collection whose type could not be inferred
    /// yet. Unifies with any type. Never written by users.
    Unknown,
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
    Function,
    Path,
    File,
}

impl Type {
    /// Resolve an annotation like `u8`, `list[int]` or `dict[str, f64]`.
    pub fn from_name(name: &str, args: Vec<Type>) -> Result<Type, String> {
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!(
                    "type '{name}' takes {n} type argument(s), got {}",
                    args.len()
                ))
            }
        };
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
            "Callable" => arity(0).map(|_| Type::Function)?,
            "list" => {
                arity(1)?;
                Type::List(Box::new(args.into_iter().next().unwrap()))
            }
            "set" => {
                arity(1)?;
                let elem = args.into_iter().next().unwrap();
                if !elem.is_hashable() {
                    return Err(format!("set element type '{elem}' is not hashable"));
                }
                Type::Set(Box::new(elem))
            }
            "dict" => {
                arity(2)?;
                let mut it = args.into_iter();
                let (k, v) = (it.next().unwrap(), it.next().unwrap());
                if !k.is_hashable() {
                    return Err(format!("dict key type '{k}' is not hashable"));
                }
                Type::Dict(Box::new(k), Box::new(v))
            }
            "tuple" => Type::Tuple(args),
            _ => return Err(format!("unknown type '{name}'")),
        };
        ty.check_depth().map_err(|e| e.to_string())?;
        Ok(ty)
    }

    pub fn depth(&self) -> usize {
        match self {
            Type::List(t) | Type::Set(t) => 1 + t.depth(),
            Type::Dict(k, v) => 1 + k.depth().max(v.depth()),
            Type::Tuple(ts) => 1 + ts.iter().map(Type::depth).max().unwrap_or(0),
            _ => 0,
        }
    }

    pub fn check_depth(&self) -> crate::Result<()> {
        if self.depth() > MAX_TYPE_DEPTH {
            Err(crate::error::err!(
                Limit,
                "type nesting exceeds the maximum depth of {MAX_TYPE_DEPTH}"
            ))
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

    /// Whether values of this type may be dict keys / set elements.
    /// Floats are excluded, as in Rust (`f64: !Hash + !Eq`).
    pub fn is_hashable(&self) -> bool {
        match self {
            Type::Unknown | Type::None | Type::Bool | Type::Str | Type::Bytes | Type::Path => true,
            t if t.is_int() => true,
            Type::Tuple(ts) => ts.iter().all(Type::is_hashable),
            _ => false,
        }
    }

    /// The most specific type compatible with both, treating `Unknown` as a
    /// wildcard. `None` if the types conflict.
    pub fn unify(&self, other: &Type) -> Option<Type> {
        Some(match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => t.clone(),
            (Type::List(a), Type::List(b)) => Type::List(Box::new(a.unify(b)?)),
            (Type::Set(a), Type::Set(b)) => Type::Set(Box::new(a.unify(b)?)),
            (Type::Dict(ka, va), Type::Dict(kb, vb)) => {
                Type::Dict(Box::new(ka.unify(kb)?), Box::new(va.unify(vb)?))
            }
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => Type::Tuple(
                a.iter()
                    .zip(b)
                    .map(|(x, y)| x.unify(y))
                    .collect::<Option<_>>()?,
            ),
            (a, b) if a == b => a.clone(),
            _ => return None,
        })
    }

    pub fn accepts(&self, actual: &Type) -> bool {
        self.unify(actual).is_some()
    }

    /// Element type yielded when iterating a value of this type, if it is
    /// iterable.
    pub fn iter_elem(&self) -> Option<Type> {
        match self {
            Type::List(t) | Type::Set(t) => Some((**t).clone()),
            Type::Dict(k, _) => Some((**k).clone()),
            Type::Str => Some(Type::Str),
            Type::Bytes => Some(Type::U8),
            Type::Tuple(ts) => {
                let mut it = ts.iter();
                let first = it.next().cloned().unwrap_or(Type::Unknown);
                it.try_fold(first, |acc, t| acc.unify(t))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unknown => write!(f, "?"),
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
            Type::Tuple(ts) => {
                write!(f, "tuple[")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, "]")
            }
            Type::Function => write!(f, "Callable"),
            Type::Path => write!(f, "Path"),
            Type::File => write!(f, "File"),
        }
    }
}

//! Exposing Rust functions to scripts.
//!
//! Any `Fn(A, B, ...) -> R` (up to 8 arguments) whose argument types
//! implement [`FromValue`] and whose return type implements [`IntoValue`] can
//! be registered with [`Interpreter::register_fn`](crate::Interpreter::register_fn).
//! The femtopython parameter types are derived from the Rust signature, so
//! calls are arity- and type-checked like annotated `def`s and numeric
//! literal arguments adopt the parameter type (`hypot(3, 4)` passes two
//! `float`s to an `Fn(f64, f64) -> f64`).
//!
//! | Rust                          | femtopython              |
//! |-------------------------------|--------------------------|
//! | `bool`, `i8`..`u64`, `f32`, `f64` | same-named primitive |
//! | `String`                      | `str`                    |
//! | [`Bytes`]                     | `bytes`                  |
//! | `PathBuf`                     | `Path`                   |
//! | `()`                          | `None` (return only)     |
//! | `Option<T>`                   | `Optional[T]`            |
//! | `Vec<T>`                      | `list[T]`                |
//! | `HashMap<K, V>`               | `dict[K, V]`             |
//! | `(A, B)` .. `(A, B, C, D)`    | `tuple[A, B, ..]`        |
//! | [`Value`]                     | any type (unchecked)     |
//! | `Result<T, E>` (return only)  | `T`, or raises `E`       |

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result, err};
use crate::types::{FnSig, Type};
use crate::value::Value;

/// A Rust type that can be received from a script.
pub trait FromValue: Sized {
    /// The femtopython type this accepts. [`Type::Unknown`] accepts anything.
    fn ty() -> Type;
    /// Convert; only called with values whose type `ty()` accepts.
    fn from_value(v: Value) -> Result<Self>;
}

/// A Rust type that can be handed back to a script.
pub trait IntoValue {
    fn ty() -> Type;
    fn into_value(self) -> Result<Value>;
}

/// `bytes`. A newtype so that `Vec<u8>` can unambiguously mean `list[u8]`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bytes(pub Vec<u8>);

fn wrong(expected: &Type, v: &Value) -> Error {
    err!(Type, "expected '{expected}', got '{}'", v.type_of())
}

macro_rules! prim {
    ($($t:ty => $v:ident),*) => {$(
        impl FromValue for $t {
            fn ty() -> Type { Type::$v }
            fn from_value(v: Value) -> Result<Self> {
                match v { Value::$v(x) => Ok(x), other => Err(wrong(&Type::$v, &other)) }
            }
        }
        impl IntoValue for $t {
            fn ty() -> Type { Type::$v }
            fn into_value(self) -> Result<Value> { Ok(Value::$v(self)) }
        }
    )*};
}
prim!(bool => Bool, i8 => I8, i16 => I16, i32 => I32, i64 => I64,
      u8 => U8, u16 => U16, u32 => U32, u64 => U64, f32 => F32, f64 => F64);

impl FromValue for Value {
    fn ty() -> Type {
        Type::Unknown
    }
    fn from_value(v: Value) -> Result<Self> {
        Ok(v)
    }
}

impl IntoValue for Value {
    fn ty() -> Type {
        Type::Unknown
    }
    fn into_value(self) -> Result<Value> {
        Ok(self)
    }
}

impl FromValue for String {
    fn ty() -> Type {
        Type::Str
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            other => Err(wrong(&Type::Str, &other)),
        }
    }
}

impl IntoValue for String {
    fn ty() -> Type {
        Type::Str
    }
    fn into_value(self) -> Result<Value> {
        Ok(Value::Str(self.into()))
    }
}

impl IntoValue for &str {
    fn ty() -> Type {
        Type::Str
    }
    fn into_value(self) -> Result<Value> {
        Ok(Value::Str(self.into()))
    }
}

impl FromValue for Bytes {
    fn ty() -> Type {
        Type::Bytes
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::Bytes(b) => Ok(Bytes(b.to_vec())),
            other => Err(wrong(&Type::Bytes, &other)),
        }
    }
}

impl IntoValue for Bytes {
    fn ty() -> Type {
        Type::Bytes
    }
    fn into_value(self) -> Result<Value> {
        Ok(Value::Bytes(self.0.into()))
    }
}

impl FromValue for PathBuf {
    fn ty() -> Type {
        Type::Path
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::Path(p) => Ok((*p).clone()),
            other => Err(wrong(&Type::Path, &other)),
        }
    }
}

impl IntoValue for PathBuf {
    fn ty() -> Type {
        Type::Path
    }
    fn into_value(self) -> Result<Value> {
        Ok(Value::Path(Arc::new(self)))
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn ty() -> Type {
        Type::optional(T::ty())
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::None => Ok(None),
            v => T::from_value(v).map(Some),
        }
    }
}

impl<T: IntoValue> IntoValue for Option<T> {
    fn ty() -> Type {
        Type::optional(T::ty())
    }
    fn into_value(self) -> Result<Value> {
        self.map_or(Ok(Value::None), T::into_value)
    }
}

impl IntoValue for () {
    fn ty() -> Type {
        Type::None
    }
    fn into_value(self) -> Result<Value> {
        Ok(Value::None)
    }
}

impl<T: FromValue> FromValue for Vec<T> {
    fn ty() -> Type {
        Type::List(Box::new(T::ty()))
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::List(l) => l.items.iter().cloned().map(T::from_value).collect(),
            other => Err(wrong(&Self::ty(), &other)),
        }
    }
}

impl<T: IntoValue> IntoValue for Vec<T> {
    fn ty() -> Type {
        Type::List(Box::new(T::ty()))
    }
    fn into_value(self) -> Result<Value> {
        let items = self.into_iter().map(T::into_value).collect::<Result<_>>()?;
        Value::new_list(T::ty(), items)
    }
}

impl<K: FromValue + Hash + Eq, V: FromValue> FromValue for HashMap<K, V> {
    fn ty() -> Type {
        Type::Dict(Box::new(K::ty()), Box::new(V::ty()))
    }
    fn from_value(v: Value) -> Result<Self> {
        match v {
            Value::Dict(d) => d
                .map
                .iter()
                .map(|(k, v)| Ok((K::from_value(k.to_value())?, V::from_value(v.clone())?)))
                .collect(),
            other => Err(wrong(&Self::ty(), &other)),
        }
    }
}

impl<K: IntoValue, V: IntoValue> IntoValue for HashMap<K, V> {
    fn ty() -> Type {
        Type::Dict(Box::new(K::ty()), Box::new(V::ty()))
    }
    fn into_value(self) -> Result<Value> {
        let entries = self
            .into_iter()
            .map(|(k, v)| Ok((k.into_value()?, v.into_value()?)))
            .collect::<Result<_>>()?;
        Value::new_dict(K::ty(), V::ty(), entries)
    }
}

macro_rules! tuple {
    ($($t:ident $v:ident),*) => {
        impl<$($t: FromValue),*> FromValue for ($($t,)*) {
            fn ty() -> Type { Type::Tuple(vec![$($t::ty()),*]) }
            fn from_value(v: Value) -> Result<Self> {
                match v {
                    Value::Tuple(items) if items.len() == [$(stringify!($t)),*].len() => {
                        let mut it = items.iter().cloned();
                        $(let $v = $t::from_value(it.next().unwrap())?;)*
                        Ok(($($v,)*))
                    }
                    other => Err(wrong(&Self::ty(), &other)),
                }
            }
        }
        impl<$($t: IntoValue),*> IntoValue for ($($t,)*) {
            fn ty() -> Type { Type::Tuple(vec![$($t::ty()),*]) }
            fn into_value(self) -> Result<Value> {
                let ($($v,)*) = self;
                Value::new_tuple(vec![$($v.into_value()?),*])
            }
        }
    };
}
tuple!(A a, B b);
tuple!(A a, B b, C c);
tuple!(A a, B b, C c, D d);

/// Errors a native function may return. `Error` keeps its kind, strings
/// become a `ValueError`, `io::Error` an `IOError`, and boxed errors pass
/// through to the caller as `ErrorKind::Native`.
pub trait IntoScriptError {
    fn into_script_error(self) -> Error;
}

impl IntoScriptError for Error {
    fn into_script_error(self) -> Error {
        self
    }
}

impl IntoScriptError for String {
    fn into_script_error(self) -> Error {
        Error::new(ErrorKind::Value(self))
    }
}

impl IntoScriptError for &str {
    fn into_script_error(self) -> Error {
        Error::new(ErrorKind::Value(self.into()))
    }
}

impl IntoScriptError for std::io::Error {
    fn into_script_error(self) -> Error {
        Error::io("I/O error in a native function", self)
    }
}

/// Any other host error: aborts the script and reaches the caller as
/// [`ErrorKind::Native`], downcastable with
/// [`Error::downcast_native`].
impl IntoScriptError for Box<dyn std::error::Error + Send + Sync> {
    fn into_script_error(self) -> Error {
        Error::new(ErrorKind::Native(self.into()))
    }
}

impl<T: IntoValue, E: IntoScriptError> IntoValue for std::result::Result<T, E> {
    fn ty() -> Type {
        T::ty()
    }
    fn into_value(self) -> Result<Value> {
        self.map_err(E::into_script_error)?.into_value()
    }
}

type RawFn = dyn Fn(Vec<Value>) -> Result<Value> + Send + Sync;

/// A host function callable from scripts.
pub struct NativeFn {
    pub name: String,
    /// Parameter types, or `None` for an unchecked variadic raw function.
    params: Option<Vec<Type>>,
    ret: Type,
    func: Box<RawFn>,
}

impl fmt::Debug for NativeFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<native function {}>", self.name)
    }
}

impl NativeFn {
    /// An unchecked function receiving the raw argument values.
    pub fn raw(
        name: impl Into<String>,
        func: impl Fn(&[Value]) -> Result<Value> + Send + Sync + 'static,
    ) -> NativeFn {
        NativeFn {
            name: name.into(),
            params: None,
            ret: Type::Unknown,
            func: Box::new(move |args| func(&args)),
        }
    }

    pub fn params(&self) -> Option<&[Type]> {
        self.params.as_deref()
    }

    pub fn ret(&self) -> &Type {
        &self.ret
    }

    /// `Callable[[A, ...], R]`, or bare `Callable` for raw functions.
    pub fn signature(&self) -> Type {
        Type::Function(self.params.as_ref().map(|params| {
            Box::new(FnSig {
                params: params.clone(),
                ret: self.ret.clone(),
            })
        }))
    }

    pub(crate) fn call(&self, args: Vec<Value>) -> Result<Value> {
        if let Some(params) = &self.params {
            if args.len() != params.len() {
                return Err(err!(
                    Type,
                    "{}() takes {} argument(s) but {} were given",
                    self.name,
                    params.len(),
                    args.len()
                ));
            }
            for (i, (p, a)) in params.iter().zip(&args).enumerate() {
                if !p.accepts(&a.type_of()) {
                    return Err(err!(
                        Type,
                        "{}() argument {} must be '{p}', not '{}'",
                        self.name,
                        i + 1,
                        a.type_of()
                    ));
                }
            }
        }
        (self.func)(args)
    }
}

/// Implemented for Rust functions and closures that can be exposed to
/// scripts. `Marker` only disambiguates the blanket impls per arity.
pub trait IntoNativeFn<Marker> {
    fn into_native_fn(self, name: String) -> NativeFn;
}

macro_rules! native_arity {
    ($($t:ident $v:ident),*) => {
        impl<F, R, $($t),*> IntoNativeFn<(R, $($t,)*)> for F
        where
            F: Fn($($t),*) -> R + Send + Sync + 'static,
            R: IntoValue,
            $($t: FromValue,)*
        {
            fn into_native_fn(self, name: String) -> NativeFn {
                NativeFn {
                    name,
                    params: Some(vec![$($t::ty()),*]),
                    ret: R::ty(),
                    #[allow(unused_mut, unused_variables)]
                    func: Box::new(move |args: Vec<Value>| {
                        let mut it = args.into_iter();
                        $(let $v = $t::from_value(it.next().expect("arity checked"))?;)*
                        (self)($($v),*).into_value()
                    }),
                }
            }
        }
    };
}
native_arity!();
native_arity!(A a);
native_arity!(A a, B b);
native_arity!(A a, B b, C c);
native_arity!(A a, B b, C c, D d);
native_arity!(A a, B b, C c, D d, E e);
native_arity!(A a, B b, C c, D d, E e, G g);
native_arity!(A a, B b, C c, D d, E e, G g, H h);
native_arity!(A a, B b, C c, D d, E e, G g, H h, I i);

//! Operator semantics. These mirror Rust's primitive operators: both
//! operands must have the same type, integer arithmetic is checked (overflow
//! is an error, as in a debug build), `/` on integers truncates, and `%` is
//! Rust's remainder (sign follows the dividend). `//` is floor division and
//! `**` exponentiation, as in Python.

use std::cmp::Ordering;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::ast::{BinOp, CmpOp, UnOp};
use crate::error::{Error, Result, err};
use crate::types::Type;
use crate::value::{IterView, Key, Set, Value, range_len};

trait Int: Copy + Ord + std::fmt::Display {
    const ZERO: Self;
    const BITS: u32;
    fn c_add(self, o: Self) -> Option<Self>;
    fn c_sub(self, o: Self) -> Option<Self>;
    fn c_mul(self, o: Self) -> Option<Self>;
    fn c_div(self, o: Self) -> Option<Self>;
    fn c_rem(self, o: Self) -> Option<Self>;
    fn c_shl(self, n: u32) -> Option<Self>;
    fn c_shr(self, n: u32) -> Option<Self>;
    fn c_pow(self, n: u32) -> Option<Self>;
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
    fn is_negative(self) -> bool;
    fn minus_one(self) -> Option<Self>;
}

macro_rules! impl_int {
    ($($t:ty),*) => {$(
        impl Int for $t {
            const ZERO: Self = 0;
            const BITS: u32 = <$t>::BITS;
            fn c_add(self, o: Self) -> Option<Self> { self.checked_add(o) }
            fn c_sub(self, o: Self) -> Option<Self> { self.checked_sub(o) }
            fn c_mul(self, o: Self) -> Option<Self> { self.checked_mul(o) }
            fn c_div(self, o: Self) -> Option<Self> { self.checked_div(o) }
            fn c_rem(self, o: Self) -> Option<Self> { self.checked_rem(o) }
            fn c_shl(self, n: u32) -> Option<Self> { self.checked_shl(n) }
            fn c_shr(self, n: u32) -> Option<Self> { self.checked_shr(n) }
            fn c_pow(self, n: u32) -> Option<Self> { self.checked_pow(n) }
            fn and(self, o: Self) -> Self { self & o }
            fn or(self, o: Self) -> Self { self | o }
            fn xor(self, o: Self) -> Self { self ^ o }
            #[allow(unused_comparisons)]
            fn is_negative(self) -> bool { self < 0 }
            fn minus_one(self) -> Option<Self> { self.checked_sub(1) }
        }
    )*};
}
impl_int!(i8, i16, i32, i64, u8, u16, u32, u64);

fn int_op<T: Int>(op: BinOp, a: T, b: T, ty: &Type) -> Result<T> {
    let overflow = || err!(Overflow, "{ty} overflow in {a} {} {b}", op.symbol());
    match op {
        BinOp::Add => a.c_add(b).ok_or_else(overflow),
        BinOp::Sub => a.c_sub(b).ok_or_else(overflow),
        BinOp::Mul => a.c_mul(b).ok_or_else(overflow),
        BinOp::Div | BinOp::Mod | BinOp::FloorDiv if b == T::ZERO => {
            Err(err!(ZeroDivision, "division by zero"))
        }
        BinOp::Div => a.c_div(b).ok_or_else(overflow),
        BinOp::Mod => a.c_rem(b).ok_or_else(overflow),
        BinOp::FloorDiv => {
            let q = a.c_div(b).ok_or_else(overflow)?;
            let r = a.c_rem(b).ok_or_else(overflow)?;
            if r != T::ZERO && (r.is_negative() != b.is_negative()) {
                q.minus_one().ok_or_else(overflow)
            } else {
                Ok(q)
            }
        }
        BinOp::BitAnd => Ok(a.and(b)),
        BinOp::BitOr => Ok(a.or(b)),
        BinOp::BitXor => Ok(a.xor(b)),
        BinOp::Shl | BinOp::Shr | BinOp::Pow => unreachable!("handled by int_amount_op"),
    }
}

/// Operators whose right operand may be any integer type: shifts and `**`.
fn int_amount_op<T: Int>(op: BinOp, a: T, amount: &Value, ty: &Type) -> Result<T> {
    let n = amount.as_i128().expect("caller checked");
    if op == BinOp::Pow {
        let n = u32::try_from(n).map_err(|_| {
            err!(
                Value,
                "integer exponent must be between 0 and {}, got {n}",
                u32::MAX
            )
        })?;
        return a
            .c_pow(n)
            .ok_or_else(|| err!(Overflow, "{ty} overflow in {a} ** {n}"));
    }
    let n = u32::try_from(n)
        .ok()
        .filter(|n| *n < T::BITS)
        .ok_or_else(|| err!(Overflow, "shift amount {n} out of range for {ty}"))?;
    Ok(match op {
        BinOp::Shl => a.c_shl(n),
        _ => a.c_shr(n),
    }
    .expect("amount checked against BITS"))
}

macro_rules! float_op {
    ($op:expr, $a:expr, $b:expr, $ty:expr) => {{
        let (a, b) = ($a, $b);
        Ok(match $op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Mod => a % b,
            BinOp::FloorDiv => (a / b).floor(),
            BinOp::Pow => a.powf(b),
            op => {
                return Err(err!(
                    Type,
                    "unsupported operand types for {}: '{}' and '{}'",
                    op.symbol(),
                    $ty,
                    $ty
                ));
            }
        })
    }};
}

fn mismatch(op: &str, a: &Value, b: &Value) -> Error {
    let (ta, tb) = (a.type_of(), b.type_of());
    if ta != tb && (ta.is_int() || ta.is_float()) && (tb.is_int() || tb.is_float()) {
        err!(
            Type,
            "mismatched types for {op}: '{ta}' and '{tb}' (convert one side explicitly, e.g. {tb}(x))"
        )
    } else {
        err!(
            Type,
            "unsupported operand types for {op}: '{ta}' and '{tb}'"
        )
    }
}

/// Whether the right operand of `op` may be any integer type (for integer
/// left operands).
pub fn takes_int_amount(op: BinOp) -> bool {
    matches!(op, BinOp::Shl | BinOp::Shr | BinOp::Pow)
}

fn set_op(op: BinOp, x: &Set, y: &Set, a: &Value, b: &Value) -> Result<Value> {
    let elem = x
        .elem
        .unify(&y.elem)
        .ok_or_else(|| mismatch(op.symbol(), a, b))?;
    let items: IndexSet<Key> = match op {
        BinOp::BitOr => x.items.union(&y.items).cloned().collect(),
        BinOp::BitAnd => x.items.intersection(&y.items).cloned().collect(),
        BinOp::Sub => x.items.difference(&y.items).cloned().collect(),
        BinOp::BitXor => x.items.symmetric_difference(&y.items).cloned().collect(),
        _ => return Err(mismatch(op.symbol(), a, b)),
    };
    Ok(Value::Set(Arc::new(Set { elem, items })))
}

/// Apply a binary operator. `max_len` bounds the size of concatenated
/// strings/bytes.
pub fn binary(op: BinOp, a: &Value, b: &Value, max_len: usize) -> Result<Value> {
    use Value as V;
    macro_rules! ints {
        ($($v:ident),*) => {
            match (op, a, b) {
                $(
                    (op, V::$v(x), _) if takes_int_amount(op) && b.as_i128().is_some() =>
                        return Ok(V::$v(int_amount_op(op, *x, b, &Type::$v)?)),
                    (op, V::$v(x), V::$v(y)) if !takes_int_amount(op) =>
                        return Ok(V::$v(int_op(op, *x, *y, &Type::$v)?)),
                )*
                _ => {}
            }
        };
    }
    ints!(I8, I16, I32, I64, U8, U16, U32, U64);

    let concat_limit = |len: usize| {
        if len > max_len {
            Err(err!(
                Limit,
                "result of concatenation exceeds {max_len} bytes"
            ))
        } else {
            Ok(())
        }
    };
    match (op, a, b) {
        (_, V::F64(x), V::F64(y)) => Ok(V::F64(float_op!(op, *x, *y, Type::F64)?)),
        (_, V::F32(x), V::F32(y)) => Ok(V::F32(float_op!(op, *x, *y, Type::F32)?)),
        (BinOp::BitAnd, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x & y)),
        (BinOp::BitOr, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x | y)),
        (BinOp::BitXor, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x ^ y)),
        (_, V::Set(x), V::Set(y)) => set_op(op, x, y, a, b),
        (BinOp::Add, V::Str(x), V::Str(y)) => {
            concat_limit(x.len() + y.len())?;
            Ok(V::Str(format!("{x}{y}").into()))
        }
        (BinOp::Add, V::Bytes(x), V::Bytes(y)) => {
            concat_limit(x.len() + y.len())?;
            Ok(V::Bytes([&x[..], &y[..]].concat().into()))
        }
        (BinOp::Div, V::Path(p), V::Str(s)) => Ok(V::Path(Arc::new(p.join(&**s)))),
        (BinOp::Div, V::Path(p), V::Path(q)) => Ok(V::Path(Arc::new(p.join(&**q)))),
        _ => Err(mismatch(op.symbol(), a, b)),
    }
}

pub fn unary(op: UnOp, v: &Value) -> Result<Value> {
    use Value as V;
    let bad = || {
        err!(
            Type,
            "bad operand type for unary {}: '{}'",
            unop_symbol(op),
            v.type_of()
        )
    };
    let overflow = || err!(Overflow, "{} overflow in negation", v.type_of());
    Ok(match (op, v) {
        (UnOp::Not, V::Bool(b)) => V::Bool(!b),
        (UnOp::Not, _) => {
            return Err(err!(
                Type,
                "'not' requires a bool, got '{}' (there is no implicit truthiness)",
                v.type_of()
            ));
        }
        (UnOp::Pos, v) if v.as_i128().is_some() || matches!(v, V::F32(_) | V::F64(_)) => v.clone(),
        (UnOp::Neg, V::I8(x)) => V::I8(x.checked_neg().ok_or_else(overflow)?),
        (UnOp::Neg, V::I16(x)) => V::I16(x.checked_neg().ok_or_else(overflow)?),
        (UnOp::Neg, V::I32(x)) => V::I32(x.checked_neg().ok_or_else(overflow)?),
        (UnOp::Neg, V::I64(x)) => V::I64(x.checked_neg().ok_or_else(overflow)?),
        (UnOp::Neg, V::F32(x)) => V::F32(-x),
        (UnOp::Neg, V::F64(x)) => V::F64(-x),
        (UnOp::Neg, V::U8(_) | V::U16(_) | V::U32(_) | V::U64(_)) => {
            return Err(err!(Type, "cannot negate unsigned type '{}'", v.type_of()));
        }
        (UnOp::Invert, V::I8(x)) => V::I8(!x),
        (UnOp::Invert, V::I16(x)) => V::I16(!x),
        (UnOp::Invert, V::I32(x)) => V::I32(!x),
        (UnOp::Invert, V::I64(x)) => V::I64(!x),
        (UnOp::Invert, V::U8(x)) => V::U8(!x),
        (UnOp::Invert, V::U16(x)) => V::U16(!x),
        (UnOp::Invert, V::U32(x)) => V::U32(!x),
        (UnOp::Invert, V::U64(x)) => V::U64(!x),
        _ => return Err(bad()),
    })
}

fn unop_symbol(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Pos => "+",
        UnOp::Invert => "~",
        UnOp::Not => "not",
    }
}

fn comparable(op: &str, a: &Value, b: &Value) -> Result<()> {
    let (ta, tb) = (a.type_of(), b.type_of());
    if ta.accepts(&tb) || tb.accepts(&ta) {
        Ok(())
    } else {
        Err(err!(Type, "cannot compare '{ta}' with '{tb}' using {op}"))
    }
}

/// Structural equality. `None` equals only `None` (and may be compared with
/// anything, for `Optional` values). Otherwise errors if the types differ.
pub fn values_equal(a: &Value, b: &Value) -> Result<bool> {
    use Value as V;
    if matches!(a, V::None) || matches!(b, V::None) {
        return Ok(matches!((a, b), (V::None, V::None)));
    }
    comparable("==", a, b)?;
    Ok(match (a, b) {
        (V::F32(x), V::F32(y)) => x == y,
        (V::F64(x), V::F64(y)) => x == y,
        (V::Bool(x), V::Bool(y)) => x == y,
        (V::Str(x), V::Str(y)) => x == y,
        (V::Bytes(x), V::Bytes(y)) => x == y,
        (V::Path(x), V::Path(y)) => x == y,
        (V::List(x), V::List(y)) => {
            x.items.len() == y.items.len() && all_equal(&x.items, &y.items)?
        }
        (V::Tuple(x), V::Tuple(y)) => x.len() == y.len() && all_equal(x, y)?,
        (V::Set(x), V::Set(y)) => x.items == y.items,
        (V::Dict(x), V::Dict(y)) => {
            if x.map.len() != y.map.len() {
                return Ok(false);
            }
            for (k, v) in &x.map {
                match y.map.get(k) {
                    Some(w) if values_equal(v, w)? => {}
                    _ => return Ok(false),
                }
            }
            true
        }
        (a, b) if a.as_i128().is_some() => a.as_i128() == b.as_i128(),
        _ => {
            return Err(err!(
                Type,
                "values of type '{}' cannot be compared",
                a.type_of()
            ));
        }
    })
}

fn all_equal(a: &[Value], b: &[Value]) -> Result<bool> {
    for (x, y) in a.iter().zip(b) {
        if !values_equal(x, y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Ordering, as Rust's `PartialOrd`. `None` for unordered floats (NaN) and
/// for sets that aren't subsets of one another.
fn partial_cmp(a: &Value, b: &Value) -> Result<Option<Ordering>> {
    use Value as V;
    comparable("an ordering comparison", a, b)?;
    Ok(match (a, b) {
        (V::F32(x), V::F32(y)) => x.partial_cmp(y),
        (V::F64(x), V::F64(y)) => x.partial_cmp(y),
        (V::Bool(x), V::Bool(y)) => Some(x.cmp(y)),
        (V::Str(x), V::Str(y)) => Some(x.cmp(y)),
        (V::Bytes(x), V::Bytes(y)) => Some(x.cmp(y)),
        (V::List(x), V::List(y)) => seq_cmp(&x.items, &y.items)?,
        (V::Tuple(x), V::Tuple(y)) => seq_cmp(x, y)?,
        // Subset ordering, as in Python.
        (V::Set(x), V::Set(y)) => {
            match (x.items.is_subset(&y.items), y.items.is_subset(&x.items)) {
                (true, true) => Some(Ordering::Equal),
                (true, false) => Some(Ordering::Less),
                (false, true) => Some(Ordering::Greater),
                (false, false) => None,
            }
        }
        (a, b) if a.as_i128().is_some() => a.as_i128().partial_cmp(&b.as_i128()),
        _ => {
            return Err(err!(
                Type,
                "values of type '{}' are not ordered",
                a.type_of()
            ));
        }
    })
}

fn seq_cmp(a: &[Value], b: &[Value]) -> Result<Option<Ordering>> {
    for (x, y) in a.iter().zip(b) {
        match partial_cmp(x, y)? {
            Some(Ordering::Equal) => {}
            other => return Ok(other),
        }
    }
    Ok(Some(a.len().cmp(&b.len())))
}

/// Comparison operators other than `in`/`not in`.
pub fn compare(op: CmpOp, a: &Value, b: &Value) -> Result<bool> {
    Ok(match op {
        CmpOp::Eq => values_equal(a, b)?,
        CmpOp::Ne => !values_equal(a, b)?,
        CmpOp::Lt => partial_cmp(a, b)? == Some(Ordering::Less),
        CmpOp::Le => matches!(partial_cmp(a, b)?, Some(Ordering::Less | Ordering::Equal)),
        CmpOp::Gt => partial_cmp(a, b)? == Some(Ordering::Greater),
        CmpOp::Ge => matches!(
            partial_cmp(a, b)?,
            Some(Ordering::Greater | Ordering::Equal)
        ),
        CmpOp::In | CmpOp::NotIn => unreachable!("membership is handled by the interpreter"),
    })
}

fn check_member_type(elem: &Type, item: &Value) -> Result<()> {
    if elem.accepts(&item.type_of()) || item.type_of().accepts(elem) {
        Ok(())
    } else {
        Err(err!(
            Type,
            "cannot test '{}' for membership in a collection of '{elem}'",
            item.type_of()
        ))
    }
}

pub(crate) fn seq_contains<'a>(
    items: impl IntoIterator<Item = &'a Value>,
    item: &Value,
) -> Result<bool> {
    for v in items {
        if values_equal(v, item)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `item in container`. `Ok(None)` means the container is a lazy view that
/// the interpreter must iterate (it may call script functions).
pub fn contains(container: &Value, item: &Value) -> Result<Option<bool>> {
    use Value as V;
    let hashed = |elem: &Type| -> Result<Key> {
        check_member_type(elem, item)?;
        Key::from_value(item)
    };
    Ok(Some(match container {
        V::List(l) => {
            check_member_type(&l.elem, item)?;
            seq_contains(&l.items, item)?
        }
        V::Tuple(t) => seq_contains(t.iter(), item)?,
        V::Set(s) => s.items.contains(&hashed(&s.elem)?),
        V::Dict(d) => d.map.contains_key(&hashed(&d.key)?),
        V::Str(s) => match item {
            V::Str(sub) => s.contains(&**sub),
            _ => {
                return Err(err!(
                    Type,
                    "'in <str>' requires str, not '{}'",
                    item.type_of()
                ));
            }
        },
        V::Bytes(b) => match item {
            V::U8(x) => b.contains(x),
            V::Bytes(sub) => sub.is_empty() || b.windows(sub.len()).any(|w| w == &sub[..]),
            _ => {
                return Err(err!(
                    Type,
                    "'in <bytes>' requires u8 or bytes, not '{}'",
                    item.type_of()
                ));
            }
        },
        V::Iter(view) => match &**view {
            IterView::Range {
                start,
                stop,
                step,
                ty,
            } => {
                check_member_type(ty, item)?;
                let Some(n) = item.as_i128() else {
                    return Ok(Some(false));
                };
                let len = range_len(*start, *stop, *step) as i128;
                let offset = n - start;
                offset % step == 0 && (0..len).contains(&(offset / step))
            }
            IterView::Keys(d) => d.map.contains_key(&hashed(&d.key)?),
            IterView::Values(d) => seq_contains(d.map.values(), item)?,
            IterView::Items(d) => match item {
                V::Tuple(kv) if kv.len() == 2 => match d.map.get(&Key::from_value(&kv[0])?) {
                    Some(v) => values_equal(v, &kv[1])?,
                    None => false,
                },
                _ => false,
            },
            IterView::Map { .. } => return Ok(None),
        },
        other => return Err(err!(Type, "'{}' is not a container", other.type_of())),
    }))
}

fn int_range(t: &Type) -> (i128, i128) {
    match t {
        Type::I8 => (i8::MIN.into(), i8::MAX.into()),
        Type::I16 => (i16::MIN.into(), i16::MAX.into()),
        Type::I32 => (i32::MIN.into(), i32::MAX.into()),
        Type::I64 => (i64::MIN.into(), i64::MAX.into()),
        Type::U8 => (0, u8::MAX.into()),
        Type::U16 => (0, u16::MAX.into()),
        Type::U32 => (0, u32::MAX.into()),
        _ => (0, u64::MAX.into()),
    }
}

fn int_to_float_is_lossless(from: &Type, to: &Type) -> bool {
    match to {
        Type::F32 => matches!(from, Type::I8 | Type::U8 | Type::I16 | Type::U16),
        Type::F64 => matches!(
            from,
            Type::I8 | Type::U8 | Type::I16 | Type::U16 | Type::I32 | Type::U32
        ),
        _ => false,
    }
}

/// The type `to(x)` produces for an `x: from`, mirroring Rust: `T` where
/// `From` exists, `Optional[T]` where only `TryFrom`/`FromStr` does, and
/// `None` (the Rust `Option`) if there is no conversion at all.
pub fn conversion_type(to: &Type, from: &Type) -> Option<Type> {
    use Type as T;
    let infallible = Some(to.clone());
    let fallible = Some(Type::optional(to.clone()));
    if to == from {
        return infallible;
    }
    match (to, from) {
        (t, T::Bool) if t.is_int() || t.is_float() => infallible,
        (t, f) if t.is_int() && f.is_int() => {
            let ((fl, fh), (tl, th)) = (int_range(f), int_range(t));
            if tl <= fl && fh <= th {
                infallible
            } else {
                fallible
            }
        }
        (t, T::Str) if t.is_int() || t.is_float() => fallible,
        (T::F64, T::F32) => infallible,
        (t, f) if int_to_float_is_lossless(f, t) => infallible,
        (T::Str, T::Bytes | T::Path) => fallible,
        (T::Str, T::Function(_)) => None,
        (T::Str, _) => infallible,
        (T::Bytes, T::Str) => infallible,
        (T::Bytes, T::List(e)) if T::U8.accepts(e) => infallible,
        (T::Path, T::Str) => infallible,
        _ => None,
    }
}

/// Explicit conversion via a type used as a function, e.g. `u8(x)`.
///
/// Allowed exactly where Rust has `From` (always succeeds), `TryFrom` or
/// `FromStr` (yields `None` on failure, so the result is `Optional[T]`), plus
/// `str(x)` for any non-function value (Rust's `Display`). There is no
/// lossy `as`-style cast: `int(2.5)` is a type error.
pub fn convert(to: &Type, v: &Value) -> Result<Value> {
    let from = v.type_of();
    if conversion_type(to, &from).is_none() {
        let mut e = err!(Type, "no conversion from '{from}' to '{to}'");
        if from.is_float() && to.is_int() {
            e.message
                .push_str(" (float to integer conversion is lossy and not supported)");
        } else if from.is_int() && to.is_float() {
            e.message.push_str(" (it would lose precision)");
        }
        return Err(e);
    }
    if *to == from {
        return Ok(v.clone());
    }
    Ok(match (to, v) {
        (t, Value::Bool(b)) if t.is_int() => Value::int_of_type(t, i128::from(*b)).unwrap(),
        (t, v) if t.is_int() && v.as_i128().is_some() => {
            Value::int_of_type(t, v.as_i128().unwrap()).unwrap_or(Value::None)
        }
        (t, Value::Str(s)) if t.is_int() => {
            macro_rules! parse_int {
                ($($ty:ident => $p:ty),*) => {
                    match t {
                        $(Type::$ty => s.parse::<$p>().map_or(Value::None, Value::$ty),)*
                        _ => unreachable!(),
                    }
                };
            }
            parse_int!(I8 => i8, I16 => i16, I32 => i32, I64 => i64,
                       U8 => u8, U16 => u16, U32 => u32, U64 => u64)
        }
        (Type::F64, Value::F32(x)) => Value::F64(f64::from(*x)),
        (Type::F64, Value::Bool(b)) => Value::F64(f64::from(*b)),
        (Type::F32, Value::Bool(b)) => Value::F32(f32::from(*b)),
        (Type::F64, Value::Str(s)) => s.parse().map_or(Value::None, Value::F64),
        (Type::F32, Value::Str(s)) => s.parse().map_or(Value::None, Value::F32),
        (t, v) if t.is_float() && v.as_i128().is_some() => {
            // Exact: only reached for lossless int -> float conversions.
            let n = v.as_i128().unwrap() as f64;
            match t {
                Type::F32 => Value::F32(n as f32),
                _ => Value::F64(n),
            }
        }
        (Type::Str, Value::Bytes(b)) => std::str::from_utf8(b).map_or(Value::None, Value::str),
        (Type::Str, Value::Path(p)) => p.to_str().map_or(Value::None, Value::str),
        (Type::Str, v) => Value::Str(v.display().into()),
        (Type::Bytes, Value::Str(s)) => Value::Bytes(s.as_bytes().into()),
        (Type::Bytes, Value::List(l)) => Value::Bytes(
            l.items
                .iter()
                .map(u8::try_from)
                .collect::<Result<Vec<u8>>>()?
                .into(),
        ),
        (Type::Path, Value::Str(s)) => Value::Path(Arc::new(s.as_ref().into())),
        _ => unreachable!("conversion_type allowed it"),
    })
}

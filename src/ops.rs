//! Operator semantics. These mirror Rust's primitive operators: both
//! operands must have the same type, integer arithmetic is checked (overflow
//! is an error, as in a debug build), `/` on integers truncates, and `%` is
//! Rust's remainder (sign follows the dividend).

use std::cmp::Ordering;
use std::rc::Rc;

use crate::ast::{BinOp, CmpOp, UnOp};
use crate::error::{Error, Result, err};
use crate::types::Type;
use crate::value::Value;

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
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
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
            fn and(self, o: Self) -> Self { self & o }
            fn or(self, o: Self) -> Self { self | o }
            fn xor(self, o: Self) -> Self { self ^ o }
        }
    )*};
}
impl_int!(i8, i16, i32, i64, u8, u16, u32, u64);

fn int_op<T: Int>(op: BinOp, a: T, b: T, ty: &str) -> Result<T> {
    let overflow = || err!(Overflow, "{ty} overflow in {a} {} {b}", op.symbol());
    match op {
        BinOp::Add => a.c_add(b).ok_or_else(overflow),
        BinOp::Sub => a.c_sub(b).ok_or_else(overflow),
        BinOp::Mul => a.c_mul(b).ok_or_else(overflow),
        BinOp::Div | BinOp::Mod if b == T::ZERO => Err(err!(ZeroDivision, "division by zero")),
        BinOp::Div => a.c_div(b).ok_or_else(overflow),
        BinOp::Mod => a.c_rem(b).ok_or_else(overflow),
        BinOp::BitAnd => Ok(a.and(b)),
        BinOp::BitOr => Ok(a.or(b)),
        BinOp::BitXor => Ok(a.xor(b)),
        BinOp::Shl | BinOp::Shr => unreachable!("shifts are handled separately"),
    }
}

fn shift<T: Int>(op: BinOp, a: T, amount: &Value, ty: &str) -> Result<T> {
    let n = amount.as_i128().ok_or_else(|| {
        err!(
            Type,
            "shift amount must be an integer, not '{}'",
            amount.type_of()
        )
    })?;
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

fn float_op<T>(op: BinOp, a: T, b: T, ty: &Type) -> Result<T>
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Rem<Output = T>,
{
    Ok(match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        _ => {
            return Err(err!(
                Type,
                "unsupported operand types for {}: '{ty}' and '{ty}'",
                op.symbol()
            ));
        }
    })
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

/// Apply a binary operator. `max_len` bounds the size of concatenated
/// strings/bytes.
pub fn binary(op: BinOp, a: &Value, b: &Value, max_len: usize) -> Result<Value> {
    use Value as V;
    macro_rules! ints {
        ($($v:ident),*) => {
            match (op, a, b) {
                $(
                    (BinOp::Shl | BinOp::Shr, V::$v(x), _) if b.as_i128().is_some() =>
                        return Ok(V::$v(shift(op, *x, b, stringify!($v))?)),
                    (_, V::$v(x), V::$v(y)) if !matches!(op, BinOp::Shl | BinOp::Shr) =>
                        return Ok(V::$v(int_op(op, *x, *y, &a.type_of().to_string())?)),
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
        (_, V::F64(x), V::F64(y)) => Ok(V::F64(float_op(op, *x, *y, &Type::F64)?)),
        (_, V::F32(x), V::F32(y)) => Ok(V::F32(float_op(op, *x, *y, &Type::F32)?)),
        (BinOp::BitAnd, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x & y)),
        (BinOp::BitOr, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x | y)),
        (BinOp::BitXor, V::Bool(x), V::Bool(y)) => Ok(V::Bool(x ^ y)),
        (BinOp::Add, V::Str(x), V::Str(y)) => {
            concat_limit(x.len() + y.len())?;
            Ok(V::Str(format!("{x}{y}").into()))
        }
        (BinOp::Add, V::Bytes(x), V::Bytes(y)) => {
            concat_limit(x.len() + y.len())?;
            Ok(V::Bytes([&x[..], &y[..]].concat().into()))
        }
        (BinOp::Div, V::Path(p), V::Str(s)) => Ok(V::Path(Rc::new(p.join(&**s)))),
        (BinOp::Div, V::Path(p), V::Path(q)) => Ok(V::Path(Rc::new(p.join(&**q)))),
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

fn same_type(op: &str, a: &Value, b: &Value) -> Result<()> {
    if a.type_of().accepts(&b.type_of()) {
        Ok(())
    } else {
        Err(err!(
            Type,
            "cannot compare '{}' with '{}' using {op}",
            a.type_of(),
            b.type_of()
        ))
    }
}

/// Structural equality. Errors if the types differ or aren't comparable.
pub fn values_equal(a: &Value, b: &Value) -> Result<bool> {
    use Value as V;
    same_type("==", a, b)?;
    Ok(match (a, b) {
        (V::None, V::None) => true,
        (V::F32(x), V::F32(y)) => x == y,
        (V::F64(x), V::F64(y)) => x == y,
        (V::Bool(x), V::Bool(y)) => x == y,
        (V::Str(x), V::Str(y)) => x == y,
        (V::Bytes(x), V::Bytes(y)) => x == y,
        (V::Path(x), V::Path(y)) => x == y,
        (V::List(x), V::List(y)) => {
            x.items.len() == y.items.len() && all_equal(&x.items, &y.items)?
        }
        (V::Tuple(x), V::Tuple(y)) => all_equal(x, y)?,
        (V::Set(x), V::Set(y)) => x.items == y.items,
        (V::Dict(x), V::Dict(y)) => {
            x.map.len() == y.map.len()
                && x.map.iter().try_fold(true, |acc, (k, v)| {
                    Ok::<_, Error>(acc && y.map.get(k).map_or(Ok(false), |w| values_equal(v, w))?)
                })?
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

/// Ordering, as Rust's `PartialOrd`. `None` for unordered floats (NaN).
fn partial_cmp(a: &Value, b: &Value) -> Result<Option<Ordering>> {
    use Value as V;
    same_type("an ordering comparison", a, b)?;
    Ok(match (a, b) {
        (V::F32(x), V::F32(y)) => x.partial_cmp(y),
        (V::F64(x), V::F64(y)) => x.partial_cmp(y),
        (V::Bool(x), V::Bool(y)) => Some(x.cmp(y)),
        (V::Str(x), V::Str(y)) => Some(x.cmp(y)),
        (V::Bytes(x), V::Bytes(y)) => Some(x.cmp(y)),
        (V::List(x), V::List(y)) => seq_cmp(&x.items, &y.items)?,
        (V::Tuple(x), V::Tuple(y)) => seq_cmp(x, y)?,
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
    })
}

/// Can `from` be converted to the float type `to` losslessly, i.e. does Rust
/// implement `From<from> for to`?
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

/// Explicit conversion via a type used as a function, e.g. `u8(x)`.
///
/// Allowed exactly where Rust has `From`/`TryFrom` (checked at runtime) or
/// `FromStr` (parsing a `str`), plus `str(x)` for any non-function value
/// (Rust's `Display`).
pub fn convert(to: &Type, v: &Value) -> Result<Value> {
    let from = v.type_of();
    if *to == from {
        return Ok(v.clone());
    }
    let no_conversion = || {
        let mut e = err!(Type, "no conversion from '{from}' to '{to}'");
        if from.is_float() && to.is_int() {
            e.message
                .push_str(" (float to integer conversion is lossy and not supported)");
        } else if from.is_int() && to.is_float() {
            e.message.push_str(" (it would lose precision)");
        }
        e
    };
    let parse_err = |s: &str| err!(Value, "invalid literal for {to}: {s:?}");
    Ok(match (to, v) {
        (t, Value::Bool(b)) if t.is_int() => Value::int_of_type(t, i128::from(*b)).unwrap(),
        (t, v) if t.is_int() && v.as_i128().is_some() => {
            let n = v.as_i128().unwrap();
            Value::int_of_type(t, n).ok_or_else(|| err!(Overflow, "{n} is out of range for {t}"))?
        }
        (t, Value::Str(s)) if t.is_int() => {
            macro_rules! parse_int {
                ($($ty:ident => $p:ty),*) => {
                    match t {
                        $(Type::$ty => Value::$ty(s.parse::<$p>().map_err(|_| parse_err(s))?),)*
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
        (t, v) if int_to_float_is_lossless(&from, t) => {
            let n = v.as_i128().unwrap() as f64; // exact: |n| < 2**32
            match t {
                Type::F32 => Value::F32(n as f32), // exact: |n| < 2**16
                _ => Value::F64(n),
            }
        }
        (Type::F64, Value::Str(s)) => Value::F64(s.parse().map_err(|_| parse_err(s))?),
        (Type::F32, Value::Str(s)) => Value::F32(s.parse().map_err(|_| parse_err(s))?),
        (Type::Str, Value::Bytes(b)) => Value::Str(
            std::str::from_utf8(b)
                .map_err(|e| err!(Value, "bytes are not valid UTF-8: {e}"))?
                .into(),
        ),
        (Type::Str, Value::Path(p)) => Value::Str(
            p.to_str()
                .ok_or_else(|| err!(Value, "path is not valid UTF-8"))?
                .into(),
        ),
        (Type::Str, v) if !v.is_callable() => Value::Str(v.display().into()),
        (Type::Bytes, Value::Str(s)) => Value::Bytes(s.as_bytes().into()),
        (Type::Bytes, Value::List(l)) if l.elem.accepts(&Type::U8) => Value::Bytes(
            l.items
                .iter()
                .map(u8::try_from)
                .collect::<Result<Vec<u8>>>()?
                .into(),
        ),
        (Type::Path, Value::Str(s)) => Value::Path(Rc::new(s.as_ref().into())),
        _ => return Err(no_conversion()),
    })
}

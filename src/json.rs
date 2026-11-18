// Copyright (C) 2026 jade
// SPDX-License-Identifier: GPL-3.0-only

//! `json.loads` / `json.dumps`.
//!
//! The parser is a strict recognizer for RFC 8259 JSON: no comments, no
//! trailing commas, no NaN/Infinity, no lone surrogates. Nesting is bounded
//! by [`MAX_TYPE_DEPTH`], so recursion depth is bounded regardless of input.
//! Output matches Python's default `json.dumps` formatting (`", "` and
//! `": "` separators, non-ASCII escaped).

use std::fmt::Write as _;

use crate::error::{Error, Result, err};
use crate::types::{MAX_TYPE_DEPTH, Type};
use crate::value::{Key, Value};

/// Containers deeper than this can't be represented as values.
const MAX_JSON_DEPTH: usize = MAX_TYPE_DEPTH - 1;

pub fn loads(text: &str) -> Result<Value> {
    let mut p = Parser {
        s: text.as_bytes(),
        pos: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.pos != p.s.len() {
        return Err(p.error("extra data after the JSON value"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, msg: &str) -> Error {
        err!(Value, "invalid JSON at offset {}: {msg}", self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", c as char)))
        }
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value> {
        if self.s[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(self.error("unexpected character"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'n') => self.literal("null", Value::None),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'"') => Ok(Value::Str(self.string()?.into())),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn check_depth(&self, depth: usize) -> Result<()> {
        if depth > MAX_JSON_DEPTH {
            Err(Error::limit(crate::LimitExceeded::NestingDepth(
                MAX_TYPE_DEPTH,
            )))
        } else {
            Ok(())
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.check_depth(depth)?;
        self.expect(b'[')?;
        self.ws();
        let mut items = Vec::new();
        if !self.eat(b']') {
            loop {
                self.ws();
                items.push(self.value(depth)?);
                self.ws();
                if self.eat(b']') {
                    break;
                }
                self.expect(b',')?;
            }
        }
        // Homogeneous arrays get a precise element type; mixed ones (which
        // are fine in JSON) become list[Any].
        let elem = items
            .iter()
            .try_fold(Type::Unknown, |acc, v| acc.unify(&v.type_of()))
            .unwrap_or(Type::Any);
        Value::new_list(elem, items)
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.check_depth(depth)?;
        self.expect(b'{')?;
        self.ws();
        let mut entries = Vec::new();
        if !self.eat(b'}') {
            loop {
                self.ws();
                if self.peek() != Some(b'"') {
                    return Err(self.error("object keys must be strings"));
                }
                let k = Value::Str(self.string()?.into());
                self.ws();
                self.expect(b':')?;
                self.ws();
                entries.push((k, self.value(depth)?));
                self.ws();
                if self.eat(b'}') {
                    break;
                }
                self.expect(b',')?;
            }
        }
        // Value types widen to Any as needed (see `Value::new_dict`); later
        // duplicate keys overwrite earlier ones, as in Python.
        Value::new_dict(Type::Str, Type::Unknown, entries)
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        self.eat(b'-');
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.error("invalid number")),
        }
        let mut is_float = false;
        if self.eat(b'.') {
            is_float = true;
            if self.digits() == 0 {
                return Err(self.error("expected digits after '.'"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            if self.digits() == 0 {
                return Err(self.error("expected digits in exponent"));
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.pos]).expect("ASCII");
        if is_float {
            Ok(Value::F64(text.parse().expect("validated float syntax")))
        } else {
            text.parse()
                .map(Value::I64)
                .map_err(|_| err!(Value, "JSON integer {text} is out of range for int"))
        }
    }

    fn hex4(&mut self) -> Result<u16> {
        let digits = self
            .s
            .get(self.pos..self.pos + 4)
            .filter(|d| d.iter().all(u8::is_ascii_hexdigit))
            .ok_or_else(|| self.error("expected 4 hex digits after \\u"))?;
        self.pos += 4;
        Ok(u16::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let start = self.pos;
            // Copy the run of plain characters in one go. The input came
            // from a `&str`, and we only split at ASCII bytes, so every run
            // is valid UTF-8.
            while self
                .peek()
                .is_some_and(|c| c != b'"' && c != b'\\' && c >= 0x20)
            {
                self.pos += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.pos]).expect("utf-8 input"));
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let Some(c) = self.peek() else {
                        return Err(self.error("unterminated string"));
                    };
                    self.pos += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                if !(self.eat(b'\\') && self.eat(b'u')) {
                                    return Err(self.error("unpaired surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.error("unpaired surrogate"));
                                }
                                0x10000
                                    + ((u32::from(hi) - 0xD800) << 10)
                                    + (u32::from(lo) - 0xDC00)
                            } else {
                                u32::from(hi)
                            };
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| self.error("unpaired surrogate"))?,
                            );
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                Some(_) => return Err(self.error("control character in string")),
            }
        }
    }
}

/// Serialize `v`. Output longer than `max_len` bytes is an error.
pub fn dumps(v: &Value, max_len: usize) -> Result<String> {
    let mut out = String::new();
    write_value(v, &mut out, max_len)?;
    Ok(out)
}

fn check_len(out: &str, max_len: usize) -> Result<()> {
    if out.len() > max_len {
        Err(Error::limit(crate::LimitExceeded::Size(max_len)))
    } else {
        Ok(())
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_float(x: f64, debug: String, out: &mut String) -> Result<()> {
    if !x.is_finite() {
        return Err(err!(
            Value,
            "out of range float values are not JSON compliant: {x}"
        ));
    }
    out.push_str(&debug);
    Ok(())
}

fn write_key(k: &Key, out: &mut String) -> Result<()> {
    match k {
        Key::Str(s) => write_str(s, out),
        Key::Bool(b) => out.push_str(if *b { "\"true\"" } else { "\"false\"" }),
        Key::None => out.push_str("\"null\""),
        k => match k.to_value().as_i128() {
            Some(n) => write!(out, "\"{n}\"").unwrap(),
            None => {
                return Err(err!(
                    Type,
                    "JSON keys must be str, int, bool or None, not '{}'",
                    k.to_value().type_of()
                ));
            }
        },
    }
    Ok(())
}

fn write_seq<'a>(
    items: impl Iterator<Item = &'a Value>,
    out: &mut String,
    max_len: usize,
) -> Result<()> {
    out.push('[');
    for (i, v) in items.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_value(v, out, max_len)?;
    }
    out.push(']');
    Ok(())
}

fn write_value(v: &Value, out: &mut String, max_len: usize) -> Result<()> {
    match v {
        Value::None => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::F32(x) => write_float(f64::from(*x), format!("{x:?}"), out)?,
        Value::F64(x) => write_float(*x, format!("{x:?}"), out)?,
        Value::Str(s) => write_str(s, out),
        Value::List(l) => write_seq(l.items.iter(), out, max_len)?,
        Value::Tuple(items) => write_seq(items.iter(), out, max_len)?,
        Value::Dict(d) => {
            out.push('{');
            for (i, (k, v)) in d.map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_key(k, out)?;
                out.push_str(": ");
                write_value(v, out, max_len)?;
            }
            out.push('}');
        }
        v if v.as_i128().is_some() => write!(out, "{}", v.as_i128().unwrap()).unwrap(),
        other => {
            return Err(err!(
                Type,
                "object of type '{}' is not JSON serializable",
                other.type_of()
            ));
        }
    }
    check_len(out, max_len)
}

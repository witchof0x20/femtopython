//! Hand-written lexer. The token language is regular apart from indentation,
//! which is tracked with an explicit, bounded stack and turned into
//! `Indent`/`Dedent` tokens so the grammar itself stays context-free.

use std::fmt;
use std::sync::Arc;

use crate::error::{Error, Result};

/// Maximum number of nested indentation levels.
pub const MAX_INDENT_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Newline,
    Indent,
    Dedent,
    Name(String),
    Int(u64),
    Float(f64),
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
    // Keywords
    Def,
    Return,
    Lambda,
    And,
    Or,
    Not,
    True,
    False,
    None,
    Pass,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Is,
    Break,
    Continue,
    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Arrow,
    Plus,
    Minus,
    Star,
    Slash,
    SlashSlash,
    StarStar,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,
    Lt,
    Le,
    EqEq,
    Ne,
    Ge,
    Gt,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    PercentEq,
    SlashEq,
    SlashSlashEq,
    StarStarEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Tok::Newline => "newline",
            Tok::Indent => "indent",
            Tok::Dedent => "dedent",
            Tok::Name(n) => return write!(f, "name '{n}'"),
            Tok::Int(i) => return write!(f, "integer {i}"),
            Tok::Float(x) => return write!(f, "float {x:?}"),
            Tok::Str(_) => "string literal",
            Tok::Bytes(_) => "bytes literal",
            Tok::Def => "'def'",
            Tok::Return => "'return'",
            Tok::Lambda => "'lambda'",
            Tok::And => "'and'",
            Tok::Or => "'or'",
            Tok::Not => "'not'",
            Tok::True => "'True'",
            Tok::False => "'False'",
            Tok::None => "'None'",
            Tok::Pass => "'pass'",
            Tok::If => "'if'",
            Tok::Elif => "'elif'",
            Tok::Else => "'else'",
            Tok::While => "'while'",
            Tok::For => "'for'",
            Tok::In => "'in'",
            Tok::Is => "'is'",
            Tok::Break => "'break'",
            Tok::Continue => "'continue'",
            Tok::LParen => "'('",
            Tok::RParen => "')'",
            Tok::LBracket => "'['",
            Tok::RBracket => "']'",
            Tok::LBrace => "'{'",
            Tok::RBrace => "'}'",
            Tok::Comma => "','",
            Tok::Colon => "':'",
            Tok::Dot => "'.'",
            Tok::Arrow => "'->'",
            Tok::Plus => "'+'",
            Tok::Minus => "'-'",
            Tok::Star => "'*'",
            Tok::Slash => "'/'",
            Tok::SlashSlash => "'//'",
            Tok::StarStar => "'**'",
            Tok::Percent => "'%'",
            Tok::Amp => "'&'",
            Tok::Pipe => "'|'",
            Tok::Caret => "'^'",
            Tok::Tilde => "'~'",
            Tok::Shl => "'<<'",
            Tok::Shr => "'>>'",
            Tok::Lt => "'<'",
            Tok::Le => "'<='",
            Tok::EqEq => "'=='",
            Tok::Ne => "'!='",
            Tok::Ge => "'>='",
            Tok::Gt => "'>'",
            Tok::Assign => "'='",
            Tok::PlusEq => "'+='",
            Tok::MinusEq => "'-='",
            Tok::StarEq => "'*='",
            Tok::PercentEq => "'%='",
            Tok::SlashEq => "'/='",
            Tok::SlashSlashEq => "'//='",
            Tok::StarStarEq => "'**='",
            Tok::AmpEq => "'&='",
            Tok::PipeEq => "'|='",
            Tok::CaretEq => "'^='",
            Tok::ShlEq => "'<<='",
            Tok::ShrEq => "'>>='",
        };
        f.write_str(s)
    }
}

/// Python keywords that femtopython deliberately does not support. They are
/// reserved so that e.g. `if x:` gives a clear error instead of a confusing
/// parse failure, and so they can be added later without breaking scripts.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "as", "assert", "async", "await", "break", "class", "continue", "del", "elif", "else",
    "except", "finally", "for", "from", "global", "if", "import", "in", "is", "nonlocal", "raise",
    "try", "while", "with", "yield",
];

pub type Spanned = (usize, Tok, usize);

pub fn lex(src: &str) -> Result<Vec<Spanned>> {
    Lexer {
        src,
        bytes: src.as_bytes(),
        pos: 0,
        toks: Vec::new(),
        indents: vec![0],
        bracket_depth: 0,
    }
    .run()
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    toks: Vec<Spanned>,
    indents: Vec<usize>,
    bracket_depth: usize,
}

impl Lexer<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, n: usize) -> Option<u8> {
        self.bytes.get(self.pos + n).copied()
    }

    fn push(&mut self, start: usize, tok: Tok) {
        self.toks.push((start, tok, self.pos));
    }

    fn error(&self, start: usize, msg: impl Into<String>) -> Error {
        Error::new(crate::ErrorKind::Syntax, msg).at((start, self.pos.max(start + 1)))
    }

    fn run(mut self) -> Result<Vec<Spanned>> {
        let mut at_line_start = true;
        loop {
            if at_line_start && self.bracket_depth == 0 {
                at_line_start = false;
                if !self.indentation()? {
                    break;
                }
            }
            let Some(c) = self.peek() else { break };
            let start = self.pos;
            match c {
                b' ' | b'\t' => self.pos += 1,
                b'\r' if self.peek_at(1) == Some(b'\n') => self.pos += 1,
                b'\n' => {
                    self.pos += 1;
                    if self.bracket_depth == 0 {
                        self.toks.push((start, Tok::Newline, start + 1));
                        at_line_start = true;
                    }
                }
                b'#' => self.skip_comment(),
                b'0'..=b'9' => self.number()?,
                b'.' if self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => self.number()?,
                b'"' | b'\'' => {
                    let s = self.string(false)?;
                    let s = String::from_utf8(s).expect("str literal is built from chars");
                    self.push(start, Tok::Str(s.into()));
                }
                b'b' | b'B' if matches!(self.peek_at(1), Some(b'"' | b'\'')) => {
                    self.pos += 1;
                    let b = self.string(true)?;
                    self.push(start, Tok::Bytes(b.into()));
                }
                c if c == b'_' || c.is_ascii_alphabetic() => self.word()?,
                _ => self.punct()?,
            }
        }
        if self.bracket_depth > 0 {
            // Leave the statement unterminated so the parser reports
            // "unexpected end of input", which a REPL treats as incomplete.
            return Ok(self.toks);
        }
        if self.toks.last().is_some_and(|(_, t, _)| *t != Tok::Newline) {
            self.toks.push((self.pos, Tok::Newline, self.pos));
        }
        while self.indents.len() > 1 {
            self.indents.pop();
            self.toks.push((self.pos, Tok::Dedent, self.pos));
        }
        Ok(self.toks)
    }

    fn skip_comment(&mut self) {
        while self.peek().is_some_and(|c| c != b'\n') {
            self.pos += 1;
        }
    }

    /// Handle leading whitespace on a logical line. Blank and comment-only
    /// lines are skipped entirely. Returns `false` at end of input.
    fn indentation(&mut self) -> Result<bool> {
        loop {
            let line_start = self.pos;
            let mut col = 0;
            while let Some(c) = self.peek() {
                match c {
                    b' ' => col += 1,
                    b'\t' => {
                        return Err(self.error(self.pos, "tabs are not allowed in indentation"));
                    }
                    _ => break,
                }
                self.pos += 1;
            }
            match self.peek() {
                None => return Ok(false),
                Some(b'#') => self.skip_comment(),
                Some(b'\r') if self.peek_at(1) == Some(b'\n') => self.pos += 1,
                Some(b'\n') => {}
                Some(_) => {
                    let top = *self.indents.last().unwrap();
                    if col > top {
                        if self.indents.len() > MAX_INDENT_DEPTH {
                            return Err(self.error(line_start, "too many levels of indentation"));
                        }
                        self.indents.push(col);
                        self.toks.push((line_start, Tok::Indent, self.pos));
                    } else {
                        while col < *self.indents.last().unwrap() {
                            self.indents.pop();
                            self.toks.push((self.pos, Tok::Dedent, self.pos));
                        }
                        if col != *self.indents.last().unwrap() {
                            return Err(self.error(
                                line_start,
                                "unindent does not match any outer indentation level",
                            ));
                        }
                    }
                    return Ok(true);
                }
            }
            // Blank or comment-only line: consume its newline and retry.
            if self.peek() == Some(b'\n') {
                self.pos += 1;
            }
        }
    }

    fn word(&mut self) -> Result<()> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c == b'_' || c.is_ascii_alphanumeric())
        {
            self.pos += 1;
        }
        let word = &self.src[start..self.pos];
        let tok = match word {
            "def" => Tok::Def,
            "return" => Tok::Return,
            "lambda" => Tok::Lambda,
            "and" => Tok::And,
            "or" => Tok::Or,
            "not" => Tok::Not,
            "True" => Tok::True,
            "False" => Tok::False,
            "None" => Tok::None,
            "pass" => Tok::Pass,
            "if" => Tok::If,
            "elif" => Tok::Elif,
            "else" => Tok::Else,
            "while" => Tok::While,
            "for" => Tok::For,
            "in" => Tok::In,
            "is" => Tok::Is,
            "break" => Tok::Break,
            "continue" => Tok::Continue,
            w if UNSUPPORTED_KEYWORDS.contains(&w) => {
                return Err(self.error(start, format!("'{w}' is not supported by femtopython")));
            }
            w => Tok::Name(w.to_owned()),
        };
        self.push(start, tok);
        Ok(())
    }

    fn digits(&mut self, valid: impl Fn(u8) -> bool) {
        while self.peek().is_some_and(|c| c == b'_' || valid(c)) {
            self.pos += 1;
        }
    }

    fn number(&mut self) -> Result<()> {
        let start = self.pos;
        let radix = match (self.peek(), self.peek_at(1)) {
            (Some(b'0'), Some(b'x' | b'X')) => 16,
            (Some(b'0'), Some(b'o' | b'O')) => 8,
            (Some(b'0'), Some(b'b' | b'B')) => 2,
            _ => 10,
        };
        let tok = if radix != 10 {
            self.pos += 2;
            let digits_start = self.pos;
            self.digits(|c| c.is_ascii_digit() || c.is_ascii_hexdigit());
            let text = self.src[digits_start..self.pos].replace('_', "");
            u64::from_str_radix(&text, radix)
                .map(Tok::Int)
                .map_err(|e| self.error(start, format!("invalid integer literal: {e}")))?
        } else {
            let mut is_float = false;
            self.digits(|c| c.is_ascii_digit());
            if self.peek() == Some(b'.') && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
                is_float = true;
                self.pos += 1;
                self.digits(|c| c.is_ascii_digit());
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                let save = self.pos;
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    is_float = true;
                    self.digits(|c| c.is_ascii_digit());
                } else {
                    self.pos = save;
                }
            }
            let raw = &self.src[start..self.pos];
            if raw.contains("__") || raw.ends_with('_') || raw.contains("_.") || raw.contains("._")
            {
                return Err(self.error(start, "invalid underscore placement in number"));
            }
            let text = raw.replace('_', "");
            if is_float {
                Tok::Float(
                    text.parse()
                        .map_err(|e| self.error(start, format!("invalid float literal: {e}")))?,
                )
            } else {
                if text.len() > 1 && text.starts_with('0') && text.bytes().any(|b| b != b'0') {
                    return Err(self.error(
                        start,
                        "leading zeros are not allowed in decimal integer literals; use 0o for octal",
                    ));
                }
                Tok::Int(text.parse().map_err(|_| {
                    self.error(start, "integer literal is too large (maximum is 2**64 - 1)")
                })?)
            }
        };
        if self
            .peek()
            .is_some_and(|c| c == b'_' || c.is_ascii_alphanumeric())
        {
            self.pos += 1;
            return Err(self.error(start, "invalid number literal"));
        }
        self.push(start, tok);
        Ok(())
    }

    /// Lex a single-line quoted literal. For `bytes`, only ASCII characters
    /// and `\xHH` escapes are allowed.
    fn string(&mut self, bytes: bool) -> Result<Vec<u8>> {
        let start = if bytes { self.pos - 1 } else { self.pos };
        let quote = self.peek().unwrap();
        self.pos += 1;
        let mut out = Vec::new();
        loop {
            let Some(c) = self.src[self.pos..].chars().next() else {
                return Err(self.error(start, "unterminated string literal"));
            };
            let c_start = self.pos;
            self.pos += c.len_utf8();
            match c {
                '\n' | '\r' => {
                    self.pos = c_start;
                    return Err(self.error(start, "unterminated string literal"));
                }
                c if c as u32 == quote as u32 => return Ok(out),
                '\\' => self.escape(bytes, &mut out)?,
                c if bytes && !c.is_ascii() => {
                    return Err(self.error(c_start, "bytes literals may only contain ASCII"));
                }
                c => {
                    let mut buf = [0; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    fn hex(&mut self, n: usize, esc_start: usize) -> Result<u32> {
        let end = self.pos + n;
        let digits = self.src.get(self.pos..end).unwrap_or("");
        if digits.len() != n || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(self.error(esc_start, format!("expected {n} hex digits in escape")));
        }
        self.pos = end;
        Ok(u32::from_str_radix(digits, 16).unwrap())
    }

    fn escape(&mut self, bytes: bool, out: &mut Vec<u8>) -> Result<()> {
        let esc_start = self.pos - 1;
        let Some(c) = self.peek() else {
            return Err(self.error(esc_start, "unterminated string literal"));
        };
        self.pos += 1;
        let simple = match c {
            b'\\' => Some(b'\\'),
            b'\'' => Some(b'\''),
            b'"' => Some(b'"'),
            b'n' => Some(b'\n'),
            b'r' => Some(b'\r'),
            b't' => Some(b'\t'),
            b'0' => Some(b'\0'),
            _ => None,
        };
        if let Some(b) = simple {
            out.push(b);
            return Ok(());
        }
        let code = match c {
            b'x' => self.hex(2, esc_start)?,
            b'u' if !bytes => self.hex(4, esc_start)?,
            b'U' if !bytes => self.hex(8, esc_start)?,
            _ => {
                self.pos = esc_start + 1;
                return Err(self.error(esc_start, "invalid escape sequence"));
            }
        };
        if bytes {
            out.push(code as u8);
        } else {
            let ch = char::from_u32(code)
                .ok_or_else(|| self.error(esc_start, "escape is not a valid Unicode scalar"))?;
            let mut buf = [0; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
        Ok(())
    }

    fn punct(&mut self) -> Result<()> {
        let start = self.pos;
        let three = (
            self.peek().unwrap(),
            self.peek_at(1).unwrap_or(0),
            self.peek_at(2).unwrap_or(0),
        );
        let (tok, len) = match three {
            (b'/', b'/', b'=') => (Tok::SlashSlashEq, 3),
            (b'*', b'*', b'=') => (Tok::StarStarEq, 3),
            (b'<', b'<', b'=') => (Tok::ShlEq, 3),
            (b'>', b'>', b'=') => (Tok::ShrEq, 3),
            (b'/', b'/', _) => (Tok::SlashSlash, 2),
            (b'*', b'*', _) => (Tok::StarStar, 2),
            (b'-', b'>', _) => (Tok::Arrow, 2),
            (b'<', b'<', _) => (Tok::Shl, 2),
            (b'>', b'>', _) => (Tok::Shr, 2),
            (b'<', b'=', _) => (Tok::Le, 2),
            (b'>', b'=', _) => (Tok::Ge, 2),
            (b'=', b'=', _) => (Tok::EqEq, 2),
            (b'!', b'=', _) => (Tok::Ne, 2),
            (b'+', b'=', _) => (Tok::PlusEq, 2),
            (b'-', b'=', _) => (Tok::MinusEq, 2),
            (b'*', b'=', _) => (Tok::StarEq, 2),
            (b'/', b'=', _) => (Tok::SlashEq, 2),
            (b'%', b'=', _) => (Tok::PercentEq, 2),
            (b'&', b'=', _) => (Tok::AmpEq, 2),
            (b'|', b'=', _) => (Tok::PipeEq, 2),
            (b'^', b'=', _) => (Tok::CaretEq, 2),
            (b'(', _, _) => (Tok::LParen, 1),
            (b')', _, _) => (Tok::RParen, 1),
            (b'[', _, _) => (Tok::LBracket, 1),
            (b']', _, _) => (Tok::RBracket, 1),
            (b'{', _, _) => (Tok::LBrace, 1),
            (b'}', _, _) => (Tok::RBrace, 1),
            (b',', _, _) => (Tok::Comma, 1),
            (b':', _, _) => (Tok::Colon, 1),
            (b'.', _, _) => (Tok::Dot, 1),
            (b'+', _, _) => (Tok::Plus, 1),
            (b'-', _, _) => (Tok::Minus, 1),
            (b'*', _, _) => (Tok::Star, 1),
            (b'/', _, _) => (Tok::Slash, 1),
            (b'%', _, _) => (Tok::Percent, 1),
            (b'&', _, _) => (Tok::Amp, 1),
            (b'|', _, _) => (Tok::Pipe, 1),
            (b'^', _, _) => (Tok::Caret, 1),
            (b'~', _, _) => (Tok::Tilde, 1),
            (b'<', _, _) => (Tok::Lt, 1),
            (b'>', _, _) => (Tok::Gt, 1),
            (b'=', _, _) => (Tok::Assign, 1),
            _ => {
                let c = self.src[start..].chars().next().unwrap();
                self.pos += c.len_utf8();
                return Err(self.error(start, format!("unexpected character {c:?}")));
            }
        };
        self.pos += len;
        match tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => self.bracket_depth += 1,
            Tok::RParen | Tok::RBracket | Tok::RBrace => {
                self.bracket_depth = self.bracket_depth.saturating_sub(1)
            }
            _ => {}
        }
        self.push(start, tok);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.1).collect()
    }

    #[test]
    fn indentation() {
        use Tok::*;
        assert_eq!(
            kinds("def f():\n    x\n\n    # c\n    y\nz"),
            vec![
                Def,
                Name("f".into()),
                LParen,
                RParen,
                Colon,
                Newline,
                Indent,
                Name("x".into()),
                Newline,
                Name("y".into()),
                Newline,
                Dedent,
                Name("z".into()),
                Newline
            ]
        );
    }

    #[test]
    fn brackets_join_lines() {
        use Tok::*;
        assert_eq!(
            kinds("[1,\n  2]"),
            vec![LBracket, Int(1), Comma, Int(2), RBracket, Newline]
        );
    }

    #[test]
    fn literals() {
        use Tok::*;
        assert_eq!(
            kinds(r#"0xff 1_000 1.5e3 'a\n' b"\x00z""#),
            vec![
                Int(255),
                Int(1000),
                Float(1500.0),
                Str("a\n".into()),
                Bytes(vec![0, b'z'].into()),
                Newline
            ]
        );
    }

    #[test]
    fn rejects() {
        for bad in [
            "import x",
            "\tx",
            "'abc",
            "b'é'",
            "5u8",
            "012",
            "99999999999999999999",
            "'\\q'",
            "$",
        ] {
            assert!(lex(bad).is_err(), "{bad:?} should not lex");
        }
    }

    #[test]
    fn bad_dedent() {
        assert!(lex("def f():\n    x\n  y").is_err());
    }
}

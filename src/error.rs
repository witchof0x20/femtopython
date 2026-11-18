//! Error type shared by the lexer, parser, checker and evaluator.

use std::fmt;

/// Byte-offset span `(start, end)` into the source text.
pub type Span = (usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Malformed input.
    Syntax,
    /// Input ended where more was expected (e.g. an unclosed bracket or an
    /// empty `def` body). A REPL should read another line.
    IncompleteInput,
    Type,
    Name,
    Value,
    Index,
    Key,
    ZeroDivision,
    Overflow,
    Io,
    /// The script tried to do something the host's [`Config`](crate::Config)
    /// forbids (e.g. `open` with filesystem access disabled).
    Permission,
    /// A resource limit from the [`Config`](crate::Config) was exceeded.
    Limit,
    /// The host set the interrupt flag (see `Interpreter::interrupt_handle`).
    Interrupted,
}

impl ErrorKind {
    pub fn name(self) -> &'static str {
        match self {
            ErrorKind::Syntax | ErrorKind::IncompleteInput => "SyntaxError",
            ErrorKind::Type => "TypeError",
            ErrorKind::Name => "NameError",
            ErrorKind::Value => "ValueError",
            ErrorKind::Index => "IndexError",
            ErrorKind::Key => "KeyError",
            ErrorKind::ZeroDivision => "ZeroDivisionError",
            ErrorKind::Overflow => "OverflowError",
            ErrorKind::Io => "IOError",
            ErrorKind::Permission => "PermissionError",
            ErrorKind::Limit => "LimitError",
            ErrorKind::Interrupted => "KeyboardInterrupt",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
            span: None,
        }
    }

    /// Attach a span if the error doesn't already carry a more precise one.
    pub fn at(mut self, span: Span) -> Self {
        if self.span.is_none() {
            self.span = Some(span);
        }
        self
    }

    /// Render the error with a `line:col` location and a caret under the
    /// offending source, for human consumption.
    pub fn render(&self, src: &str) -> String {
        let Some((start, end)) = self.span else {
            return self.to_string();
        };
        let start = start.min(src.len());
        let line_start = src[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = src[start..].find('\n').map_or(src.len(), |i| start + i);
        let line_no = src[..start].matches('\n').count() + 1;
        let col = src[line_start..start].chars().count() + 1;
        let line = &src[line_start..line_end];
        let width = src[start..end.clamp(start, line_end)]
            .chars()
            .count()
            .max(1);
        format!(
            "line {line_no}, col {col}: {self}\n  {line}\n  {}{}",
            " ".repeat(col - 1),
            "^".repeat(width)
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for Error {}

macro_rules! err {
    ($kind:ident, $($fmt:tt)*) => {
        $crate::error::Error::new($crate::error::ErrorKind::$kind, format!($($fmt)*))
    };
}
pub(crate) use err;

//! Errors returned to the host.
//!
//! Every failure (bad syntax, a script raising a type error, a resource
//! limit, an error returned by a host function) comes back to the caller as
//! an [`Error`]: an [`ErrorKind`] saying exactly what went wrong, plus the
//! source span it happened at. Match on [`ErrorKind`] for details, or on
//! [`Error::class`] for the Python-style exception class.

use std::fmt;
use std::sync::Arc;

use thiserror::Error as ThisError;

/// Byte-offset span `(start, end)` into the source text.
pub type Span = (usize, usize);

/// The Python-style exception class of an error, for coarse matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    Syntax,
    Type,
    Name,
    Value,
    Index,
    Key,
    ZeroDivision,
    Overflow,
    Io,
    Permission,
    Limit,
    Interrupted,
    Native,
}

impl ErrorClass {
    pub fn name(self) -> &'static str {
        match self {
            ErrorClass::Syntax => "SyntaxError",
            ErrorClass::Type => "TypeError",
            ErrorClass::Name => "NameError",
            ErrorClass::Value => "ValueError",
            ErrorClass::Index => "IndexError",
            ErrorClass::Key => "KeyError",
            ErrorClass::ZeroDivision => "ZeroDivisionError",
            ErrorClass::Overflow => "OverflowError",
            ErrorClass::Io => "IOError",
            ErrorClass::Permission => "PermissionError",
            ErrorClass::Limit => "LimitError",
            ErrorClass::Interrupted => "KeyboardInterrupt",
            ErrorClass::Native => "NativeError",
        }
    }
}

/// Which resource limit from the [`Config`](crate::Config) (or which
/// built-in bound) was exceeded.
#[derive(Debug, Clone, PartialEq, Eq, ThisError)]
pub enum LimitExceeded {
    #[error("expression nesting exceeds the maximum depth of {0}")]
    ExprDepth(u32),
    #[error("type or value nesting exceeds the maximum depth of {0}")]
    NestingDepth(usize),
    #[error("maximum evaluation depth ({0}) exceeded; is there unbounded recursion?")]
    EvalDepth(usize),
    #[error("step limit ({0}) exceeded")]
    Steps(u64),
    #[error("value would exceed the maximum size of {0} bytes")]
    Size(usize),
}

#[derive(Debug, Clone, ThisError)]
pub enum ErrorKind {
    /// Malformed input.
    #[error("SyntaxError: {0}")]
    Syntax(String),
    /// Input ended where more was expected (an unclosed bracket, an empty
    /// block). A REPL should read another line.
    #[error("SyntaxError: unexpected end of input")]
    IncompleteInput,
    #[error("TypeError: {0}")]
    Type(String),
    /// A name that is not defined.
    #[error("NameError: name '{0}' is not defined")]
    Name(String),
    #[error("ValueError: {0}")]
    Value(String),
    #[error("IndexError: {0}")]
    Index(String),
    /// A missing dict key or set element (shown as its repr).
    #[error("KeyError: {0}")]
    Key(String),
    #[error("ZeroDivisionError: division by zero")]
    ZeroDivision,
    #[error("OverflowError: {0}")]
    Overflow(String),
    #[error("IOError: {message}")]
    Io {
        message: String,
        #[source]
        source: Arc<std::io::Error>,
    },
    /// Something the host forbids, e.g. `open` without an
    /// [`Opener`](crate::Opener), a path outside an allowlist, or breaking a
    /// lock's access rules.
    #[error("PermissionError: {0}")]
    Permission(String),
    #[error("LimitError: {0}")]
    Limit(LimitExceeded),
    /// The host set the interrupt flag (see `Interpreter::interrupt_handle`).
    #[error("KeyboardInterrupt")]
    Interrupted,
    /// An error returned by a host function, passed through untouched. Use
    /// [`Error::downcast_native`] to get the original back.
    #[error("{0}")]
    Native(#[source] Arc<dyn std::error::Error + Send + Sync>),
}

impl ErrorKind {
    pub fn class(&self) -> ErrorClass {
        match self {
            ErrorKind::Syntax(_) | ErrorKind::IncompleteInput => ErrorClass::Syntax,
            ErrorKind::Type(_) => ErrorClass::Type,
            ErrorKind::Name(_) => ErrorClass::Name,
            ErrorKind::Value(_) => ErrorClass::Value,
            ErrorKind::Index(_) => ErrorClass::Index,
            ErrorKind::Key(_) => ErrorClass::Key,
            ErrorKind::ZeroDivision => ErrorClass::ZeroDivision,
            ErrorKind::Overflow(_) => ErrorClass::Overflow,
            ErrorKind::Io { .. } => ErrorClass::Io,
            ErrorKind::Permission(_) => ErrorClass::Permission,
            ErrorKind::Limit(_) => ErrorClass::Limit,
            ErrorKind::Interrupted => ErrorClass::Interrupted,
            ErrorKind::Native(_) => ErrorClass::Native,
        }
    }
}

/// An error and where in the source it happened.
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Option<Span>,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub fn new(kind: ErrorKind) -> Self {
        Error { kind, span: None }
    }

    pub fn limit(l: LimitExceeded) -> Self {
        Error::new(ErrorKind::Limit(l))
    }

    pub fn io(message: impl Into<String>, source: std::io::Error) -> Self {
        Error::new(ErrorKind::Io {
            message: message.into(),
            source: Arc::new(source),
        })
    }

    /// Wrap a host error so it reaches the caller unchanged.
    pub fn native(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Error::new(ErrorKind::Native(Arc::new(e)))
    }

    pub fn class(&self) -> ErrorClass {
        self.kind.class()
    }

    /// The host error inside an [`ErrorKind::Native`], if it is an `E`.
    pub fn downcast_native<E: std::error::Error + 'static>(&self) -> Option<&E> {
        match &self.kind {
            ErrorKind::Native(e) => e.downcast_ref::<E>(),
            _ => None,
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
        self.kind.fmt(f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.kind)
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Error::new(kind)
    }
}

impl From<LimitExceeded> for Error {
    fn from(l: LimitExceeded) -> Self {
        Error::limit(l)
    }
}

/// `err!(Type, "...", args)` builds an error whose kind carries a message.
macro_rules! err {
    ($kind:ident, $($fmt:tt)*) => {
        $crate::error::Error::new($crate::error::ErrorKind::$kind(format!($($fmt)*)))
    };
}
pub(crate) use err;

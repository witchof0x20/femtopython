//! Glue between the lexer and the lalrpop-generated parser.

use lalrpop_util::{ParseError, lalrpop_mod};

use crate::ast::{Expr, ExprKind, Stmt};
use crate::error::{Error, ErrorKind, Result, Span};
use crate::lexer::{Tok, lex};

lalrpop_mod!(
    #[allow(clippy::all, clippy::pedantic, unused)]
    grammar
);

type PErr = ParseError<usize, Tok, Error>;

pub(crate) fn mk(kind: ExprKind, l: usize, r: usize) -> Result<Expr, PErr> {
    Expr::new(kind, (l, r)).map_err(|error| ParseError::User { error })
}

pub(crate) fn user_err(msg: impl Into<String>, span: Span) -> PErr {
    ParseError::User {
        error: Error::new(ErrorKind::Syntax(msg.into())).at(span),
    }
}

fn convert(e: PErr, src_len: usize) -> Error {
    match e {
        ParseError::User { error } => error,
        ParseError::UnrecognizedEof { location, .. } => {
            Error::new(ErrorKind::IncompleteInput).at((location, location))
        }
        // The lexer closes open blocks with zero-width DEDENT tokens at the
        // end of input; tripping over one means a block body is missing
        // (more input needed), not that the input is wrong.
        ParseError::UnrecognizedToken {
            token: (l, Tok::Dedent, r),
            ..
        } if l == r && r == src_len => Error::new(ErrorKind::IncompleteInput).at((l, l)),
        ParseError::UnrecognizedToken {
            token: (l, tok, r),
            expected,
        } => {
            let mut msg = format!("unexpected {tok}");
            if !expected.is_empty() && expected.len() <= 6 {
                msg.push_str(&format!(", expected one of {}", expected.join(" ")));
            }
            Error::new(ErrorKind::Syntax(msg)).at((l, r))
        }
        ParseError::ExtraToken { token: (l, tok, r) } => {
            Error::new(ErrorKind::Syntax(format!("unexpected {tok}"))).at((l, r))
        }
        ParseError::InvalidToken { location } => {
            Error::new(ErrorKind::Syntax("invalid token".into())).at((location, location + 1))
        }
    }
}

fn tokens(src: &str) -> Result<impl Iterator<Item = Result<(usize, Tok, usize), Error>>> {
    Ok(lex(src)?.into_iter().map(Ok))
}

pub fn parse_program(src: &str) -> Result<Vec<Stmt>> {
    grammar::ProgramParser::new()
        .parse(tokens(src)?)
        .map_err(|e| convert(e, src.len()))
}

pub fn parse_expr(src: &str) -> Result<Expr> {
    grammar::SingleExprParser::new()
        .parse(tokens(src)?)
        .map_err(|e| convert(e, src.len()))
}

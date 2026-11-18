use std::collections::HashMap;
use std::path::PathBuf;

use femtopython::{Bytes, Error, ErrorKind, Interpreter, Value};

fn interp() -> Interpreter {
    let mut i = Interpreter::new();
    i.register_fn("hypot", |a: f64, b: f64| a.hypot(b));
    i.register_fn("add_u8", |a: u8, b: u8| {
        a.checked_add(b).ok_or("u8 overflow")
    });
    i.register_fn("shout", |s: String| s.to_uppercase());
    i.register_fn("sum", |xs: Vec<i64>| xs.iter().sum::<i64>());
    i.register_fn("pair", |a: i32, s: String| (a, s));
    i.register_fn("swap", |(a, b): (i64, String)| (b, a));
    i.register_fn("answer", || 42i64);
    i.register_fn("noop", |_: Value| ());
    i.register_fn("len_bytes", |b: Bytes| b.0.len() as u64);
    i.register_fn("rev_bytes", |b: Bytes| {
        Bytes(b.0.into_iter().rev().collect())
    });
    i.register_fn("ext", |p: PathBuf| {
        p.extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    i.register_fn("counts", |s: String| {
        let mut m: HashMap<String, i64> = HashMap::new();
        for w in s.split_whitespace() {
            *m.entry(w.to_owned()).or_default() += 1;
        }
        m
    });
    i.register_fn("total", |m: HashMap<String, i64>| m.values().sum::<i64>());
    i.register_fn("strict", |x: i64| -> Result<i64, Error> {
        Err(Error::new(ErrorKind::Key, format!("no {x}")))
    });
    i.register_fn("squares", |n: u32| {
        (0..n).map(|x| x * x).collect::<Vec<u32>>()
    });
    i
}

fn ok(src: &str) -> Value {
    interp()
        .run(src)
        .unwrap_or_else(|e| panic!("{}", e.render(src)))
}

fn fails(src: &str) -> Error {
    interp().run(src).expect_err(src)
}

#[test]
fn literal_arguments_adopt_rust_parameter_types() {
    assert_eq!(ok("hypot(3, 4)"), Value::F64(5.0));
    assert_eq!(ok("add_u8(200, 55)"), Value::U8(255));
    assert_eq!(ok("x: u8 = 1\nadd_u8(x, 2)"), Value::U8(3));
}

#[test]
fn argument_types_and_arity_are_checked() {
    assert_eq!(fails("hypot('3', 4)").kind, ErrorKind::Type);
    assert_eq!(fails("hypot(3)").kind, ErrorKind::Type);
    assert_eq!(fails("x = 1\nadd_u8(x, 2)").kind, ErrorKind::Type);
    assert_eq!(fails("add_u8(300, 1)").kind, ErrorKind::Overflow);
    assert_eq!(fails("sum(['a'])").kind, ErrorKind::Type);
}

#[test]
fn errors_from_rust_become_script_errors() {
    let e = fails("add_u8(200, 56)");
    assert_eq!(
        (e.kind, e.message.as_str()),
        (ErrorKind::Value, "u8 overflow")
    );
    assert_eq!(fails("strict(1)").kind, ErrorKind::Key);
    // Spans point at the call site.
    assert!(
        fails("a = 1\nstrict(1)")
            .render("a = 1\nstrict(1)")
            .starts_with("line 2")
    );
}

#[test]
fn compound_types() {
    assert_eq!(ok("shout('hi')"), Value::from("HI"));
    assert_eq!(ok("sum([1, 2, 3])"), Value::I64(6));
    assert_eq!(ok("sum([])"), Value::I64(0));
    assert_eq!(ok("pair(1, 'a')").repr(), "(1, 'a')");
    assert_eq!(ok("swap((1, 'a'))").repr(), "('a', 1)");
    assert_eq!(ok("answer()"), Value::I64(42));
    assert_eq!(ok("noop([1.5])"), Value::None);
    assert_eq!(ok("len_bytes(b'abc')"), Value::U64(3));
    assert_eq!(ok("rev_bytes(b'abc')").repr(), "b'cba'");
    assert_eq!(ok("ext(Path('a/b.txt'))"), Value::from("txt"));
    assert_eq!(ok("counts('a b a')['a']"), Value::I64(2));
    assert_eq!(ok("total({'x': 1, 'y': 2})"), Value::I64(3));
    assert_eq!(ok("squares(4)").repr(), "[0, 1, 4, 9]");
}

#[test]
fn native_functions_are_first_class() {
    assert_eq!(ok("map(shout, ['a', 'b'])").repr(), "['A', 'B']");
    // An empty map still learns its element type from the Rust signature.
    assert_eq!(ok("r: list[str] = map(shout, [])\nr").repr(), "[]");
    assert_eq!(ok("f = hypot\nf(6, 8)"), Value::F64(10.0));
    assert_eq!(ok("hypot").repr(), "<native function hypot>");
}

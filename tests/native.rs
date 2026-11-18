use std::collections::HashMap;
use std::path::PathBuf;

use femtopython::{Bytes, Error, ErrorClass, ErrorKind, Interpreter, Value};

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
        Err(Error::new(ErrorKind::Key(format!("no {x}"))))
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
    assert_eq!(fails("hypot('3', 4)").class(), ErrorClass::Type);
    assert_eq!(fails("hypot(3)").class(), ErrorClass::Type);
    assert_eq!(fails("x = 1\nadd_u8(x, 2)").class(), ErrorClass::Type);
    assert_eq!(fails("add_u8(300, 1)").class(), ErrorClass::Overflow);
    assert_eq!(fails("sum(['a'])").class(), ErrorClass::Type);
}

#[test]
fn errors_from_rust_become_script_errors() {
    let e = fails("add_u8(200, 56)");
    assert!(
        matches!(&e.kind, ErrorKind::Value(m) if m == "u8 overflow"),
        "{e}"
    );
    assert_eq!(fails("strict(1)").class(), ErrorClass::Key);
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
    assert_eq!(ok("list(map(shout, ['a', 'b']))").repr(), "['A', 'B']");
    // An empty map still learns its element type from the Rust signature.
    assert_eq!(ok("r: list[str] = list(map(shout, []))\nr").repr(), "[]");
    assert_eq!(ok("f = hypot\nf(6, 8)"), Value::F64(10.0));
    assert_eq!(ok("hypot").repr(), "<native function hypot>");
}

#[test]
fn options_map_to_optional() {
    let mut i = Interpreter::new();
    i.register_fn("find", |xs: Vec<i64>, x: i64| {
        xs.iter().position(|&y| y == x).map(|p| p as i64)
    });
    i.register_fn("or_zero", |x: Option<u8>| x.unwrap_or(0));
    assert_eq!(i.eval("find([5, 6], 6)").unwrap(), Value::I64(1));
    assert_eq!(i.eval("find([5, 6], 7)").unwrap(), Value::None);
    assert_eq!(i.eval("or_zero(None)").unwrap(), Value::U8(0));
    assert_eq!(i.eval("or_zero(9)").unwrap(), Value::U8(9));
    assert!(i.eval("or_zero('x')").is_err());
}

#[derive(Debug, thiserror::Error)]
#[error("sensor {0} is offline")]
struct SensorOffline(u32);

#[test]
fn host_errors_reach_the_caller_unchanged() {
    let mut i = Interpreter::new();
    i.register_fn(
        "read_sensor",
        |id: u32| -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
            if id == 7 {
                Err(Box::new(SensorOffline(id)))
            } else {
                Ok(1.5)
            }
        },
    );
    assert_eq!(i.eval("read_sensor(1)").unwrap(), Value::F64(1.5));
    let src = "def avg():\n    return read_sensor(1) + read_sensor(7)\navg()";
    let e = i.run(src).unwrap_err();
    assert_eq!(e.class(), ErrorClass::Native);
    let inner = e
        .downcast_native::<SensorOffline>()
        .expect("original error type");
    assert_eq!(inner.0, 7);
    // It's also the error's source(), and the span points at the call.
    assert_eq!(
        std::error::Error::source(&e).unwrap().to_string(),
        "sensor 7 is offline"
    );
    assert!(e.render(src).starts_with("line 2"), "{}", e.render(src));
    // `Error::native` wraps any error type explicitly.
    i.register_fn(
        "strict_sensor",
        |id: u32| -> Result<f64, femtopython::Error> {
            Err(femtopython::Error::native(SensorOffline(id)))
        },
    );
    let e = i.eval("strict_sensor(3)").unwrap_err();
    assert_eq!(e.downcast_native::<SensorOffline>().unwrap().0, 3);
}

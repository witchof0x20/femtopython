use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use femtopython::{Config, ErrorKind, InputStatus, Interpreter, Value, input_status};

fn run(src: &str) -> Value {
    Interpreter::new()
        .run(src)
        .unwrap_or_else(|e| panic!("{}", e.render(src)))
}

fn repr(src: &str) -> String {
    run(src).repr()
}

fn fails(src: &str) -> ErrorKind {
    match Interpreter::new().run(src) {
        Ok(v) => panic!(
            "expected {:?} to fail, got {v:?}",
            &src[..src.len().min(80)]
        ),
        Err(e) => e.kind,
    }
}

#[derive(Clone, Default)]
struct Capture(Rc<RefCell<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn output(src: &str) -> String {
    let cap = Capture::default();
    let mut i = Interpreter::new();
    i.set_output(cap.clone());
    i.run(src).unwrap_or_else(|e| panic!("{}", e.render(src)));
    String::from_utf8(cap.0.borrow().clone()).unwrap()
}

// ------------------------------------------------------------------ types

#[test]
fn literal_inference() {
    assert_eq!(run("2"), Value::I64(2));
    assert_eq!(run("2.5"), Value::F64(2.5));
    assert_eq!(run("'hi'"), Value::from("hi"));
    assert_eq!(run("b'hi'"), Value::Bytes(b"hi".as_slice().into()));
    assert_eq!(run("True"), Value::Bool(true));
}

#[test]
fn annotated_declarations() {
    assert_eq!(run("a: u8 = 200\na"), Value::U8(200));
    assert_eq!(run("a: i8 = -128\na"), Value::I8(-128));
    assert_eq!(run("a: f32 = 1.5\na"), Value::F32(1.5));
    assert_eq!(run("a: int = 2\na"), Value::I64(2));
    assert_eq!(run("a: float = 2\na"), Value::F64(2.0));
    assert_eq!(
        run("a: u64 = 18446744073709551615\na"),
        Value::U64(u64::MAX)
    );
    assert_eq!(fails("a: u8 = 256"), ErrorKind::Overflow);
    assert_eq!(fails("a: u8 = -1"), ErrorKind::Overflow);
    assert_eq!(fails("a: u8 = 'x'"), ErrorKind::Type);
    assert_eq!(fails("a = 9223372036854775808"), ErrorKind::Overflow);
}

#[test]
fn literals_adopt_context_type() {
    assert_eq!(run("x: u8 = 5\nx += 1\nx"), Value::U8(6));
    assert_eq!(run("x: u8 = 5\nx * 2"), Value::U8(10));
    assert_eq!(run("x: f32 = 1.5\nx * 2"), Value::F32(3.0));
    assert_eq!(run("2 * 3.5"), Value::F64(7.0));
    assert_eq!(run("def f(a: u16) -> u16: return a\nf(7)"), Value::U16(7));
    assert_eq!(run("x: list[u8] = [1, 2]\nx"), run("[u8(1), u8(2)]"));
}

#[test]
fn no_implicit_numeric_mixing() {
    assert_eq!(fails("x: u8 = 1\ny: u16 = 2\nx + y"), ErrorKind::Type);
    assert_eq!(fails("a = 1\nb = 2.0\na + b"), ErrorKind::Type);
    assert_eq!(fails("a = 1\na == 1.0"), ErrorKind::Type);
    // Two bare literals unify, preferring float: `2 * 3.5` is a calculator.
    assert_eq!(run("1 == 1.0"), Value::Bool(true));
    assert_eq!(fails("'a' + 1"), ErrorKind::Type);
}

#[test]
fn rust_integer_semantics() {
    assert_eq!(run("7 / 2"), Value::I64(3));
    assert_eq!(run("-7 / 2"), Value::I64(-3));
    assert_eq!(run("-7 % 3"), Value::I64(-1));
    assert_eq!(run("7.0 / 2.0"), Value::F64(3.5));
    assert_eq!(fails("x: u8 = 255\nx + 1"), ErrorKind::Overflow);
    assert_eq!(fails("x: u8 = 0\nx - 1"), ErrorKind::Overflow);
    assert_eq!(fails("1 / 0"), ErrorKind::ZeroDivision);
    assert_eq!(fails("x: u8 = 1\n-x"), ErrorKind::Type);
    assert_eq!(fails("1 << 64"), ErrorKind::Overflow);
}

#[test]
fn bitwise() {
    assert_eq!(run("6 & 3"), Value::I64(2));
    assert_eq!(run("6 | 3"), Value::I64(7));
    assert_eq!(run("6 ^ 3"), Value::I64(5));
    assert_eq!(run("~0"), Value::I64(-1));
    assert_eq!(run("x: u8 = 0\n~x"), Value::U8(255));
    assert_eq!(run("1 << 4"), Value::I64(16));
    assert_eq!(run("x: u8 = 1\nx << 7"), Value::U8(128));
    assert_eq!(run("-16 >> 2"), Value::I64(-4));
    assert_eq!(run("True ^ True"), Value::Bool(false));
}

#[test]
fn comparisons_and_logic() {
    assert_eq!(run("1 < 2"), Value::Bool(true));
    assert_eq!(run("'a' < 'b'"), Value::Bool(true));
    assert_eq!(run("[1, 2] == [1, 2]"), Value::Bool(true));
    assert_eq!(run("(1, 'a') < (1, 'b')"), Value::Bool(true));
    assert_eq!(run("1 < 2 and not 2 < 1"), Value::Bool(true));
    assert_eq!(run("False or True"), Value::Bool(true));
    // Short-circuit: the right side would be a NameError.
    assert_eq!(run("False and undefined"), Value::Bool(false));
    assert_eq!(fails("1 and True"), ErrorKind::Type);
    assert_eq!(fails("not 0"), ErrorKind::Type);
    assert_eq!(fails("1 < 2 < 3"), ErrorKind::Syntax);
}

#[test]
fn strings_and_bytes() {
    assert_eq!(run("'héllo'[1]"), Value::from("é"));
    assert_eq!(run("'héllo'[-1]"), Value::from("o"));
    assert_eq!(run("b'AB'[1]"), Value::U8(66));
    assert_eq!(run("'a' + 'b'"), Value::from("ab"));
    assert_eq!(repr("map(lambda c: c, 'hé')"), "['h', 'é']");
    assert_eq!(repr("map(lambda b: b, b'hi')"), "[104, 105]");
    assert_eq!(fails("'abc'[3]"), ErrorKind::Index);
}

#[test]
fn lists() {
    assert_eq!(repr("a = [1, 2, 3]\na[0] = 10\na"), "[10, 2, 3]");
    assert_eq!(repr("a = [[1], [2]]\na[1][0] += 5\na"), "[[1], [7]]");
    assert_eq!(fails("[1, 'a']"), ErrorKind::Type);
    assert_eq!(fails("a = [1]\na[0] = 'x'"), ErrorKind::Type);
    assert_eq!(fails("[1][5]"), ErrorKind::Index);
    // Value semantics: copying a list never aliases it.
    assert_eq!(repr("a = [1]\nb = a\nb[0] = 2\na"), "[1]");
}

#[test]
fn empty_collections() {
    assert_eq!(repr("a: list[u8] = []\na"), "[]");
    assert_eq!(run("a: list[u8] = []\na == []"), Value::Bool(true));
    assert_eq!(repr("d: dict[str, int] = {}\nd['x'] = 1\nd"), "{'x': 1}");
    assert_eq!(repr("d = {}\nd['x'] = 1\nd"), "{'x': 1}");
    assert_eq!(fails("d = {}\nd['x'] = 1\nd[2] = 1"), ErrorKind::Type);
    assert_eq!(
        fails("d: dict[str, u8] = {}\nd['x'] = 300"),
        ErrorKind::Overflow
    );
}

#[test]
fn dicts() {
    let d = "d = {'a': 1, 'b': 2}\n";
    assert_eq!(repr(&format!("{d}d.keys()")), "['a', 'b']");
    assert_eq!(repr(&format!("{d}d.values()")), "[1, 2]");
    assert_eq!(repr(&format!("{d}d.items()")), "[('a', 1), ('b', 2)]");
    assert_eq!(repr(&format!("{d}d['b'] += 40\nd['b']")), "42");
    assert_eq!(fails(&format!("{d}d['z']")), ErrorKind::Key);
    assert_eq!(fails(&format!("{d}d[1]")), ErrorKind::Type);
    assert_eq!(fails("{'a': 1, 2: 1}"), ErrorKind::Type);
    assert_eq!(fails("{1.5: 1}"), ErrorKind::Type);
}

#[test]
fn sets() {
    assert_eq!(repr("{1, 2, 2, 3}"), "{1, 2, 3}");
    assert_eq!(run("{1, 2} == {2, 1}"), Value::Bool(true));
    assert_eq!(fails("{1, 'a'}"), ErrorKind::Type);
    assert_eq!(fails("{1.0}"), ErrorKind::Type);
}

#[test]
fn tuples() {
    assert_eq!(repr("(1, 'a')"), "(1, 'a')");
    assert_eq!(repr("(1,)"), "(1,)");
    assert_eq!(run("(1, 'a')[1]"), Value::from("a"));
    assert_eq!(repr("map(lambda x: x * 2, (1, 2))"), "[2, 4]");
    assert_eq!(fails("map(lambda x: x, (1, 'a'))"), ErrorKind::Type);
    assert_eq!(fails("t = (1, 2)\nt[0] = 5"), ErrorKind::Type);
}

#[test]
fn conversions() {
    assert_eq!(run("u8(200)"), Value::U8(200));
    assert_eq!(run("x = 5\nu8(x)"), Value::U8(5));
    assert_eq!(run("x: u8 = 5\nint(x)"), Value::I64(5));
    assert_eq!(run("x: i32 = 5\nfloat(x)"), Value::F64(5.0));
    assert_eq!(run("float(2)"), Value::F64(2.0));
    assert_eq!(run("int('42')"), Value::I64(42));
    assert_eq!(run("float('2.5')"), Value::F64(2.5));
    assert_eq!(run("str(42)"), Value::from("42"));
    assert_eq!(run("str(b'hi')"), Value::from("hi"));
    assert_eq!(run("bytes('hi')"), Value::Bytes(b"hi".as_slice().into()));
    assert_eq!(run("int(True)"), Value::I64(1));
    assert_eq!(fails("x = 300\nu8(x)"), ErrorKind::Overflow);
    assert_eq!(fails("int(2.5)"), ErrorKind::Type); // no From<f64> for i64
    assert_eq!(fails("x = 5\nfloat(x)"), ErrorKind::Type); // no From<i64> for f64
    assert_eq!(fails("int('nope')"), ErrorKind::Value);
}

// ------------------------------------------------------- declarations, scope

#[test]
fn top_level_redeclaration_may_change_type() {
    assert_eq!(run("a = 1\na = 'now a str'\na"), Value::from("now a str"));
}

#[test]
fn no_redeclaration_inside_functions() {
    assert_eq!(fails("def f():\n    a = 1\n    a = 2"), ErrorKind::Syntax);
    assert_eq!(fails("def f(a):\n    a = 1"), ErrorKind::Syntax);
    assert_eq!(fails("def f(a, a): pass"), ErrorKind::Syntax);
    assert_eq!(fails("lambda x, x: x"), ErrorKind::Syntax);
    assert_eq!(
        run("def f(a):\n    a += 1\n    return a\nf(1)"),
        Value::I64(2)
    );
}

#[test]
fn function_declarations_shadow() {
    let src = "a = 1\ndef f():\n    a = 'local'\n    return a\nf() + str(a)";
    assert_eq!(run(src), Value::from("local1"));
}

#[test]
fn captured_variables_are_read_only() {
    assert_eq!(fails("a = 1\ndef f():\n    a += 1"), ErrorKind::Syntax);
    assert_eq!(
        fails("def f():\n    a = [1]\n    def g():\n        a[0] = 2"),
        ErrorKind::Syntax
    );
}

#[test]
fn return_outside_function() {
    assert_eq!(fails("return 1"), ErrorKind::Syntax);
}

// --------------------------------------------------------------- functions

#[test]
fn functions_are_first_class() {
    let src = "def twice(f, x): return f(f(x))\ntwice(lambda n: n * 3, 2)";
    assert_eq!(run(src), Value::I64(18));
}

#[test]
fn closures_capture_by_value() {
    let src = "def make(n):\n    return lambda x: x + n\nadd5 = make(5)\nadd5(1)";
    assert_eq!(run(src), Value::I64(6));
}

#[test]
fn nested_function_can_see_enclosing_function() {
    let src = "def outer(n):\n    def inner(): return outer\n    return inner()\nouter(1)";
    assert_eq!(repr(src), "<function outer>");
}

#[test]
fn call_time_type_checks() {
    assert_eq!(fails("def f(a: u8): pass\nf('x')"), ErrorKind::Type);
    assert_eq!(fails("def f() -> int: return 'x'\nf()"), ErrorKind::Type);
    assert_eq!(fails("def f() -> int: pass\nf()"), ErrorKind::Type);
    assert_eq!(fails("def f(a): pass\nf()"), ErrorKind::Type);
    assert_eq!(fails("x = 1\nx()"), ErrorKind::Type);
    // Unannotated parameters accept anything.
    assert_eq!(run("def id(a): return a\nid('s')"), Value::from("s"));
}

#[test]
fn multi_line_function() {
    let src = "
def area(w: float, h: float) -> float:
    a = w * h
    a *= 2
    return a / 2

area(3, 4.5)
";
    assert_eq!(run(src), Value::F64(13.5));
}

#[test]
fn unbounded_recursion_is_an_error_not_a_crash() {
    let src = "def f(n): return f(n + 1)\nf(0)";
    assert_eq!(fails(src), ErrorKind::Limit);
    assert_eq!(fails("(lambda f: f(f))(lambda f: f(f))"), ErrorKind::Limit);
}

// ---------------------------------------------------------------- builtins

#[test]
fn print_builtin() {
    assert_eq!(
        output("print(1, 'a', [b'x'], 2.0, None)\nprint()"),
        "1 a [b'x'] 2.0 None\n\n"
    );
    assert_eq!(fails("print(print)"), ErrorKind::Type);
    assert_eq!(run("print()"), Value::None);
}

#[test]
fn map_builtin() {
    assert_eq!(repr("map(lambda x: x * x, [1, 2, 3])"), "[1, 4, 9]");
    assert_eq!(repr("map(str, {1: 'a'})"), "['1']");
    assert_eq!(
        repr("def sq(x: int) -> int: return x * x\nmap(sq, [])"),
        "[]"
    );
    assert_eq!(
        run("def sq(x: int) -> int: return x * x\nr: list[int] = map(sq, [])\nr == []"),
        Value::Bool(true)
    );
    assert_eq!(fails("map(1, [1])"), ErrorKind::Type);
}

#[test]
fn paths_and_open() {
    assert_eq!(
        repr("Path('a') / 'b'"),
        format!("Path('a{}b')", std::path::MAIN_SEPARATOR)
    );
    assert_eq!(fails("open('/etc/hostname')"), ErrorKind::Permission);

    let dir = std::env::temp_dir().join(format!("femtopython-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("f.txt"), "contents").unwrap();
    let mut i = Interpreter::with_config(Config {
        allow_fs: true,
        ..Config::default()
    });
    i.set("dir", dir.clone());
    assert_eq!(
        i.run("open(dir / 'f.txt').read()").unwrap(),
        Value::from("contents")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn methods_must_be_called() {
    assert_eq!(fails("{1: 2}.keys"), ErrorKind::Type);
    assert_eq!(fails("[1].append(2)"), ErrorKind::Type);
}

// --------------------------------------------------------- syntax & limits

#[test]
fn unsupported_syntax_is_rejected_clearly() {
    let e = Interpreter::new().run("if x:\n    pass").unwrap_err();
    assert_eq!(e.kind, ErrorKind::Syntax);
    assert!(e.message.contains("not supported"), "{}", e.message);
    assert_eq!(fails("x = 1, 2"), ErrorKind::Syntax);
    assert_eq!(fails("x: nosuchtype = 1"), ErrorKind::Syntax);
    assert_eq!(fails("x: set[float] = {}"), ErrorKind::Syntax);
}

#[test]
fn deeply_nested_input_is_rejected() {
    // Redundant parentheses don't nest the AST, so they are harmless.
    let parens = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
    assert_eq!(run(&parens), Value::I64(1));
    let lists = format!("{}1{}", "[".repeat(10_000), "]".repeat(10_000));
    assert_eq!(fails(&lists), ErrorKind::Limit);
    let negs = format!("{}1", "-".repeat(10_000));
    assert_eq!(fails(&negs), ErrorKind::Limit);
    let long_chain = vec!["1"; 50_000].join(" + ");
    assert_eq!(fails(&long_chain), ErrorKind::Limit);
    let deep_type = format!("x: {}int{} = 1", "list[".repeat(100), "]".repeat(100));
    assert_eq!(fails(&deep_type), ErrorKind::Syntax);
}

#[test]
fn deeply_nested_values_are_rejected() {
    let mut src = String::from("a = 1\n");
    for _ in 0..100 {
        src.push_str("a = [a]\n");
    }
    assert_eq!(fails(&src), ErrorKind::Limit);
}

#[test]
fn string_growth_is_bounded() {
    let mut i = Interpreter::with_config(Config {
        max_str_len: 1024,
        ..Config::default()
    });
    let mut src = String::from("s = 'xx'\n");
    for _ in 0..20 {
        src.push_str("s = s + s\n");
    }
    assert_eq!(i.run(&src).unwrap_err().kind, ErrorKind::Limit);
}

#[test]
fn input_status_for_repl() {
    assert_eq!(input_status("1 +"), InputStatus::Invalid);
    assert_eq!(input_status("(1 +"), InputStatus::Incomplete);
    assert_eq!(input_status("def f():"), InputStatus::Incomplete);
    assert_eq!(
        input_status("def f():\n    return 1\n"),
        InputStatus::Complete
    );
    assert_eq!(input_status("x = [1,\n2]"), InputStatus::Complete);
}

#[test]
fn error_rendering_points_at_source() {
    let src = "a = 1\nb = a + 'x'";
    let e = Interpreter::new().run(src).unwrap_err();
    let r = e.render(src);
    assert!(r.starts_with("line 2, col 5: TypeError"), "{r}");
    assert!(r.contains("^^^^^^^"), "{r}");
}

// ---------------------------------------------------------------- embedding

#[test]
fn embedding_api() {
    let mut i = Interpreter::new();
    i.set("x", 20i64);
    i.register_fn("double", |args| match args {
        [Value::I64(n)] => Ok(Value::I64(n * 2)),
        _ => Err(femtopython::Error::new(ErrorKind::Type, "double(int)")),
    });
    i.run("def add(a: int, b: int) -> int: return a + b\ny = double(x) + 2")
        .unwrap();
    assert_eq!(i.get("y"), Some(&Value::I64(42)));
    let add = i.get("add").unwrap().clone();
    assert_eq!(
        i.call(&add, vec![Value::I64(1), Value::I64(2)]).unwrap(),
        Value::I64(3)
    );
    assert_eq!(i.eval("x * 2").unwrap(), Value::I64(40));
    // State persists across runs, and errors keep earlier effects.
    assert!(i.run("z = 1\nundefined_name").is_err());
    assert_eq!(i.get("z"), Some(&Value::I64(1)));
}

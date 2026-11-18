use std::io::Write;
use std::sync::{Arc, Mutex};

use femtopython::{
    Allowlist, Config, ErrorClass, ErrorKind, InputStatus, Interpreter, LockKind, Opener, Reader,
    Type, Value, input_status,
};

fn run(src: &str) -> Value {
    Interpreter::new()
        .run(src)
        .unwrap_or_else(|e| panic!("{}", e.render(src)))
}

fn repr(src: &str) -> String {
    run(src).repr()
}

fn fails(src: &str) -> ErrorClass {
    match Interpreter::new().run(src) {
        Ok(v) => panic!(
            "expected {:?} to fail, got {v:?}",
            &src[..src.len().min(80)]
        ),
        Err(e) => e.class(),
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
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
    String::from_utf8(cap.0.lock().unwrap().clone()).unwrap()
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
    assert_eq!(fails("a: u8 = 256"), ErrorClass::Overflow);
    assert_eq!(fails("a: u8 = -1"), ErrorClass::Overflow);
    assert_eq!(fails("a: u8 = 'x'"), ErrorClass::Type);
    assert_eq!(fails("a = 9223372036854775808"), ErrorClass::Overflow);
}

#[test]
fn literals_adopt_context_type() {
    assert_eq!(run("x: u8 = 5\nx += 1\nx"), Value::U8(6));
    assert_eq!(run("x: u8 = 5\nx * 2"), Value::U8(10));
    assert_eq!(run("x: f32 = 1.5\nx * 2"), Value::F32(3.0));
    assert_eq!(run("2 * 3.5"), Value::F64(7.0));
    assert_eq!(run("def f(a: u16) -> u16: return a\nf(7)"), Value::U16(7));
    assert_eq!(run("x: list[u8] = [1, 2]\nx"), run("[u8(1), u8(2)]"));
    assert_eq!(run("xs: list[u8] = []\nxs.append(1)\nxs[0]"), Value::U8(1));
    assert_eq!(run("x: u8 = 3\nx in [1, 2, 3]"), Value::Bool(true));
}

#[test]
fn no_implicit_numeric_mixing() {
    assert_eq!(fails("x: u8 = 1\ny: u16 = 2\nx + y"), ErrorClass::Type);
    assert_eq!(fails("a = 1\nb = 2.0\na + b"), ErrorClass::Type);
    assert_eq!(fails("a = 1\na == 1.0"), ErrorClass::Type);
    // Two bare literals unify, preferring float: `2 * 3.5` is a calculator.
    assert_eq!(run("1 == 1.0"), Value::Bool(true));
    assert_eq!(fails("'a' + 1"), ErrorClass::Type);
}

#[test]
fn rust_integer_semantics() {
    assert_eq!(run("7 / 2"), Value::I64(3));
    assert_eq!(run("-7 / 2"), Value::I64(-3));
    assert_eq!(run("-7 % 3"), Value::I64(-1));
    assert_eq!(run("7.0 / 2.0"), Value::F64(3.5));
    assert_eq!(fails("x: u8 = 255\nx + 1"), ErrorClass::Overflow);
    assert_eq!(fails("x: u8 = 0\nx - 1"), ErrorClass::Overflow);
    assert_eq!(fails("1 / 0"), ErrorClass::ZeroDivision);
    assert_eq!(fails("x: u8 = 1\n-x"), ErrorClass::Type);
    assert_eq!(fails("1 << 64"), ErrorClass::Overflow);
}

#[test]
fn floor_division_and_power() {
    assert_eq!(run("7 // 2"), Value::I64(3));
    assert_eq!(run("-7 // 2"), Value::I64(-4));
    assert_eq!(run("7 // -2"), Value::I64(-4));
    assert_eq!(run("-7.0 // 2.0"), Value::F64(-4.0));
    assert_eq!(run("2 ** 10"), Value::I64(1024));
    assert_eq!(run("-2 ** 2"), Value::I64(-4));
    assert_eq!(run("2 ** 3 ** 2"), Value::I64(512));
    assert_eq!(run("x: u8 = 2\nx ** 7"), Value::U8(128));
    assert_eq!(run("x = 2.0\nx ** 2"), Value::F64(4.0));
    assert_eq!(run("2 ** 0.5"), Value::F64(2f64.powf(0.5)));
    assert_eq!(fails("x: u8 = 2\nx ** 8"), ErrorClass::Overflow);
    assert_eq!(fails("2 ** -1"), ErrorClass::Value);
}

#[test]
fn all_augmented_assignments() {
    let src = "x = 100\nx /= 3\nx //= 2\nx **= 2\nx %= 100\nx &= 63\nx |= 64\nx ^= 1\nx <<= 2\nx >>= 1\nx";
    // 100/3=33, //2=16, **2=256, %100=56, &63=56, |64=120, ^1=121, <<2=484, >>1=242
    assert_eq!(run(src), Value::I64(242));
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
    assert_eq!(fails("1 and True"), ErrorClass::Type);
    assert_eq!(fails("not 0"), ErrorClass::Type);
    assert_eq!(fails("1 < 2 < 3"), ErrorClass::Syntax);
}

#[test]
fn conditional_expression() {
    assert_eq!(run("1 if True else 2"), Value::I64(1));
    assert_eq!(run("x: u8 = 1 if False else 2\nx"), Value::U8(2));
    assert_eq!(fails("1 if 1 else 2"), ErrorClass::Type);
}

#[test]
fn strings_and_bytes() {
    assert_eq!(run("'héllo'[1]"), Value::from("é"));
    assert_eq!(run("'héllo'[-1]"), Value::from("o"));
    assert_eq!(run("b'AB'[1]"), Value::U8(66));
    assert_eq!(run("'a' + 'b'"), Value::from("ab"));
    assert_eq!(run("'ell' in 'hello'"), Value::Bool(true));
    assert_eq!(run("66 in b'AB'"), Value::Bool(true));
    assert_eq!(run("len('héllo')"), Value::I64(5));
    assert_eq!(repr("list(map(lambda c: c, 'hé'))"), "['h', 'é']");
    assert_eq!(repr("list(b'hi')"), "[104, 105]");
    assert_eq!(fails("'abc'[3]"), ErrorClass::Index);
}

#[test]
fn lists() {
    assert_eq!(repr("a = [1, 2, 3]\na[0] = 10\na"), "[10, 2, 3]");
    assert_eq!(repr("a = [[1], [2]]\na[1][0] += 5\na"), "[[1], [7]]");
    assert_eq!(fails("[1, 'a']"), ErrorClass::Type);
    assert_eq!(fails("a = [1]\na[0] = 'x'"), ErrorClass::Type);
    assert_eq!(fails("[1][5]"), ErrorClass::Index);
    // Value semantics: copying a list never aliases it.
    assert_eq!(repr("a = [1]\nb = a\nb[0] = 2\na"), "[1]");
}

#[test]
fn list_methods() {
    assert_eq!(
        repr("a = [1]\na.append(2)\na.extend([3, 4])\na"),
        "[1, 2, 3, 4]"
    );
    assert_eq!(repr("a = [1, 2, 3]\nx = a.pop()\n(x, a)"), "(3, [1, 2])");
    assert_eq!(repr("a = [1, 2, 3]\na.pop(0)\na"), "[2, 3]");
    assert_eq!(repr("a = [[1]]\na[0].append(2)\na"), "[[1, 2]]");
    assert_eq!(repr("a = []\na.append('x')\na"), "['x']");
    assert_eq!(repr("a = [1]\na.extend(range(2, 4))\na"), "[1, 2, 3]");
    assert_eq!(fails("a = [1]\na.append('x')"), ErrorClass::Type);
    assert_eq!(fails("a = []\na.pop()"), ErrorClass::Index);
    assert_eq!(fails("[1].append(2)"), ErrorClass::Type);
    assert_eq!(run("len([1, 2])"), Value::I64(2));
    assert_eq!(run("2 in [1, 2]"), Value::Bool(true));
    assert_eq!(run("3 not in [1, 2]"), Value::Bool(true));
}

#[test]
fn empty_collections() {
    assert_eq!(repr("a: list[u8] = []\na"), "[]");
    assert_eq!(run("a: list[u8] = []\na == []"), Value::Bool(true));
    assert_eq!(repr("d: dict[str, int] = {}\nd['x'] = 1\nd"), "{'x': 1}");
    assert_eq!(repr("d = {}\nd['x'] = 1\nd"), "{'x': 1}");
    assert_eq!(fails("d = {}\nd['x'] = 1\nd[2] = 1"), ErrorClass::Type);
    assert_eq!(
        fails("d: dict[str, u8] = {}\nd['x'] = 300"),
        ErrorClass::Overflow
    );
}

#[test]
fn dicts() {
    let d = "d = {'a': 1, 'b': 2}\n";
    assert_eq!(repr(&format!("{d}list(d.keys())")), "['a', 'b']");
    assert_eq!(repr(&format!("{d}list(d.values())")), "[1, 2]");
    assert_eq!(repr(&format!("{d}list(d.items())")), "[('a', 1), ('b', 2)]");
    assert_eq!(repr(&format!("{d}d.keys()")), "dict_keys(['a', 'b'])");
    assert_eq!(repr(&format!("{d}d['b'] += 40\nd['b']")), "42");
    assert_eq!(repr(&format!("{d}d.get('z')")), "None");
    assert_eq!(repr(&format!("{d}d.get('z', 0)")), "0");
    assert_eq!(repr(&format!("{d}d.pop('a')\nd")), "{'b': 2}");
    assert_eq!(run(&format!("{d}'a' in d")), Value::Bool(true));
    assert_eq!(run(&format!("{d}len(d)")), Value::I64(2));
    assert_eq!(fails(&format!("{d}d['z']")), ErrorClass::Key);
    assert_eq!(fails(&format!("{d}d[1]")), ErrorClass::Type);
    assert_eq!(fails("{'a': 1, 2: 1}"), ErrorClass::Type);
    assert_eq!(fails("{1.5: 1}"), ErrorClass::Type);
}

#[test]
fn dict_values_are_heterogeneous_like_json() {
    let src = "d = {'name': 'x', 'n': 1, 'tags': ['a'], 'none': None}\nd";
    assert_eq!(
        repr(src),
        "{'name': 'x', 'n': 1, 'tags': ['a'], 'none': None}"
    );
    assert_eq!(
        run("d = {'a': 1}\nd['b'] = 'two'\nd['b']"),
        Value::from("two")
    );
    assert_eq!(run("d = {'a': 1, 'b': 'x'}\nd['a'] + 1"), Value::I64(2));
    assert_eq!(repr("{'a': 1, 'b': 2.5}"), "{'a': 1, 'b': 2.5}");
    assert_eq!(
        repr("d: dict[str, float] = {'a': 1, 'b': 2.5}\nd"),
        "{'a': 1.0, 'b': 2.5}"
    );
    // A nested JSON-ish document.
    assert_eq!(
        run(
            "doc = {'users': [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]}\ndoc['users'][1]['name']"
        ),
        Value::from("b")
    );
    // An annotated value type stays strict.
    assert_eq!(
        fails("d: dict[str, int] = {'a': 1}\nd['b'] = 'two'"),
        ErrorClass::Type
    );
    assert_eq!(
        run("d: dict[str, Any] = {'a': 1}\nd['b'] = 'two'\nd['b']"),
        Value::from("two")
    );
}

#[test]
fn sets() {
    assert_eq!(repr("{1, 2, 2, 3}"), "{1, 2, 3}");
    assert_eq!(run("{1, 2} == {2, 1}"), Value::Bool(true));
    assert_eq!(fails("{1, 'a'}"), ErrorClass::Type);
    assert_eq!(fails("{1.0}"), ErrorClass::Type);
}

#[test]
fn set_operations() {
    assert_eq!(repr("{1, 2} | {2, 3}"), "{1, 2, 3}");
    assert_eq!(repr("{1, 2} & {2, 3}"), "{2}");
    assert_eq!(repr("{1, 2} - {2, 3}"), "{1}");
    assert_eq!(repr("{1, 2} ^ {2, 3}"), "{1, 3}");
    assert_eq!(run("{1} <= {1, 2}"), Value::Bool(true));
    assert_eq!(run("{1} < {1}"), Value::Bool(false));
    assert_eq!(run("{1, 3} >= {1, 2}"), Value::Bool(false));
    assert_eq!(run("2 in {1, 2}"), Value::Bool(true));
    assert_eq!(run("len({1, 2, 2})"), Value::I64(2));
    assert_eq!(
        repr("s = {1}\ns.add(2)\ns.discard(5)\ns.remove(1)\ns"),
        "{2}"
    );
    assert_eq!(repr("s: set[u8] = set()\ns.add(1)\ns"), "{1}");
    assert_eq!(repr("set([1, 1, 2])"), "{1, 2}");
    assert_eq!(fails("s = {1}\ns.remove(2)"), ErrorClass::Key);
    assert_eq!(fails("{1} | {'a'}"), ErrorClass::Type);
}

#[test]
fn tuples() {
    assert_eq!(repr("(1, 'a')"), "(1, 'a')");
    assert_eq!(repr("(1,)"), "(1,)");
    assert_eq!(run("(1, 'a')[1]"), Value::from("a"));
    assert_eq!(repr("list(map(lambda x: x * 2, (1, 2)))"), "[2, 4]");
    assert_eq!(fails("list(map(lambda x: x, (1, 'a')))"), ErrorClass::Type);
    assert_eq!(fails("t = (1, 2)\nt[0] = 5"), ErrorClass::Type);
}

#[test]
fn optional() {
    assert_eq!(run("x: Optional[int] = None\nx is None"), Value::Bool(true));
    assert_eq!(
        run("x: Optional[int] = 5\nx is not None"),
        Value::Bool(true)
    );
    assert_eq!(run("x: Optional[u8] = 5\nx"), Value::U8(5));
    assert_eq!(repr("[1, None, 3]"), "[1, None, 3]");
    assert_eq!(run("xs = [1, None]\nxs[1] = 2\nxs[1]"), Value::I64(2));
    assert_eq!(run("None == None"), Value::Bool(true));
    assert_eq!(run("x = 1\nx == None"), Value::Bool(false));
    assert_eq!(fails("x: int = None"), ErrorClass::Type);
    assert_eq!(fails("def f(x: int): pass\nf(None)"), ErrorClass::Type);
    assert_eq!(
        run(
            "def f(x: Optional[int]) -> int:\n    if x is None:\n        return 0\n    return x\nf(None) + f(4)"
        ),
        Value::I64(4)
    );
    // Variables keep their annotated type across assignments.
    let src = "best: Optional[int] = None
for x in [3, 1, 2]:
    if best is None or x < best:
        best = x
best";
    assert_eq!(run(src), Value::I64(1));
    assert_eq!(
        run("x: Optional[int] = None\nx = 5\nx = None\nx is None"),
        Value::Bool(true)
    );
    // `is` is only for None checks.
    assert_eq!(fails("1 is 1"), ErrorClass::Syntax);
}

#[test]
fn conversions() {
    // From: always succeeds.
    assert_eq!(run("u8(200)"), Value::U8(200));
    assert_eq!(run("x: u8 = 5\nint(x)"), Value::I64(5));
    assert_eq!(run("x: i32 = 5\nfloat(x)"), Value::F64(5.0));
    assert_eq!(run("float(2)"), Value::F64(2.0));
    assert_eq!(run("str(42)"), Value::from("42"));
    assert_eq!(run("bytes('hi')"), Value::Bytes(b"hi".as_slice().into()));
    assert_eq!(run("int(True)"), Value::I64(1));
    // TryFrom / FromStr: Optional.
    assert_eq!(run("x = 5\nu8(x)"), Value::U8(5));
    assert_eq!(run("x = 300\nu8(x)"), Value::None);
    assert_eq!(run("u8(300)"), Value::None);
    assert_eq!(run("u64(18446744073709551615)"), Value::U64(u64::MAX));
    assert_eq!(run("x = -1\nu64(x)"), Value::None);
    assert_eq!(run("int('42')"), Value::I64(42));
    assert_eq!(run("int('nope')"), Value::None);
    assert_eq!(run("float('2.5')"), Value::F64(2.5));
    assert_eq!(run("str(b'hi')"), Value::from("hi"));
    assert_eq!(run("str(b'\\xff')"), Value::None);
    assert_eq!(fails("x = 300\ny: u8 = u8(x)"), ErrorClass::Type);
    assert_eq!(
        run("x = 300\ny: Optional[u8] = u8(x)\ny is None"),
        Value::Bool(true)
    );
    // No conversion at all (no `as`).
    assert_eq!(fails("int(2.5)"), ErrorClass::Type);
    assert_eq!(fails("x = 5\nfloat(x)"), ErrorClass::Type);
}

// ------------------------------------------------------------ control flow

#[test]
fn if_elif_else() {
    let src = "def sign(x: int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
[sign(5), sign(-5), sign(0)]";
    assert_eq!(repr(src), "[1, -1, 0]");
    assert_eq!(fails("if 1:\n    pass"), ErrorClass::Type);
}

#[test]
fn while_loops() {
    assert_eq!(
        run("i = 0\nwhile i < 10:\n    i = i + 1\ni"),
        Value::I64(10)
    );
    assert_eq!(
        run("i = 0\nwhile True:\n    i += 1\n    if i == 5:\n        break\ni"),
        Value::I64(5)
    );
    let src = "n = 0\ni = 0\nwhile i < 10:\n    i += 1\n    if i % 2 == 0:\n        continue\n    n += i\nn";
    assert_eq!(run(src), Value::I64(25));
}

#[test]
fn for_loops() {
    assert_eq!(
        run("total = 0\nfor i in range(5):\n    total += i\ntotal"),
        Value::I64(10)
    );
    assert_eq!(
        run("s = ''\nfor k, v in {'a': 1, 'b': 2}.items():\n    s = s + k + str(v)\ns"),
        Value::from("a1b2")
    );
    assert_eq!(
        run("n = 0\nfor x in [1, 2, 3, 4]:\n    if x == 3:\n        break\n    n += x\nn"),
        Value::I64(3)
    );
    assert_eq!(repr("list(range(5, 0, -2))"), "[5, 3, 1]");
    assert_eq!(repr("x: u8 = 3\nlist(range(x))"), "[0, 1, 2]");
    assert_eq!(repr("x: u8 = 1\nlist(range(x, 3))"), "[1, 2]");
    assert_eq!(fails("for a, b in [1, 2]:\n    pass"), ErrorClass::Type);
    assert_eq!(fails("break"), ErrorClass::Syntax);
    assert_eq!(fails("def f():\n    continue"), ErrorClass::Syntax);
}

#[test]
fn iterating_a_list_snapshots_it() {
    let src = "xs = [1, 2]\nfor x in xs:\n    xs.append(x)\nxs";
    assert_eq!(repr(src), "[1, 2, 1, 2]");
}

#[test]
fn loops_are_bounded_by_the_step_limit() {
    let mut i = Interpreter::with_config(Config {
        max_steps: Some(1000),
        ..Config::default()
    });
    assert_eq!(
        i.run("while True:\n    pass").unwrap_err().class(),
        ErrorClass::Limit
    );
    assert_eq!(
        i.run("list(range(100000))").unwrap_err().class(),
        ErrorClass::Limit
    );
}

#[test]
fn interrupts_stop_a_running_script() {
    let mut i = Interpreter::with_config(Config {
        max_steps: None,
        ..Config::default()
    });
    let flag = i.interrupt_handle();
    let t = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(
        i.run("while True:\n    pass").unwrap_err().class(),
        ErrorClass::Interrupted
    );
    t.join().unwrap();
}

// ------------------------------------------------------------------ scopes

#[test]
fn blocks_scope_declarations() {
    // A declaration inside a block disappears after it.
    assert_eq!(fails("if True:\n    y = 1\ny"), ErrorClass::Name);
    assert_eq!(fails("for i in range(3):\n    pass\ni"), ErrorClass::Name);
    // But blocks read and write enclosing variables directly.
    assert_eq!(run("x = 1\nif True:\n    x = 2\nx"), Value::I64(2));
    assert_eq!(
        run("x = [1]\nif True:\n    x.append(2)\nlen(x)"),
        Value::I64(2)
    );
    // Annotated declarations shadow instead.
    assert_eq!(
        run("x = 1\nif True:\n    x: str = 'inner'\n    print(x)\nx"),
        Value::I64(1)
    );
    // Writes to an enclosing block's variable keep its type.
    assert_eq!(fails("x = 1\nif True:\n    x = 'str'"), ErrorClass::Type);
}

#[test]
fn redeclaration_is_allowed_everywhere() {
    assert_eq!(run("a = 1\na = 'now a str'\na"), Value::from("now a str"));
    assert_eq!(
        run("def f():\n    a = 1\n    a = 'x'\n    return a\nf()"),
        Value::from("x")
    );
    assert_eq!(
        run("def f(a):\n    a = a * 2\n    return a\nf(2)"),
        Value::I64(4)
    );
}

#[test]
fn functions_copy_used_outer_variables_at_declaration() {
    // Later changes to the outer variable are not seen.
    assert_eq!(run("x = 1\ndef f(): return x\nx = 2\nf()"), Value::I64(1));
    // Writes inside affect only the function's copy.
    assert_eq!(
        run("x = 1\ndef f():\n    x += 10\n    return x\n[f(), f(), x]"),
        run("[11, 11, 1]")
    );
    // Names defined after the function are not visible to it.
    assert_eq!(
        fails("def f(): return later\nlater = 1\nf()"),
        ErrorClass::Name
    );
    // Declarations inside shadow.
    assert_eq!(
        run("a = 1\ndef f():\n    a: str = 'local'\n    return a\nf() + str(a)"),
        Value::from("local1")
    );
}

#[test]
fn closures() {
    let src = "def make(n):\n    return lambda x: x + n\nadd5 = make(5)\nadd5(1)";
    assert_eq!(run(src), Value::I64(6));
    let src = "def twice(f, x): return f(f(x))\ntwice(lambda n: n * 3, 2)";
    assert_eq!(run(src), Value::I64(18));
}

#[test]
fn recursion() {
    let src = "def fact(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nfact(10)";
    assert_eq!(run(src), Value::I64(3628800));
    let src = "def outer(n):\n    def inner(): return outer\n    return inner()\nouter(1)";
    assert_eq!(repr(src), "<function outer>");
}

// ------------------------------------------------------------------- locks

#[test]
fn rwlock_is_shared_read_write() {
    let src = "count: rwlock[int] = 0
def bump():
    count += 1
bump()
bump()
count";
    assert_eq!(run(src), Value::I64(2));
    let src = "seen: rwlock[list[str]] = []
def visit(s: str):
    seen.append(s)
visit('a')
visit('b')
seen";
    assert_eq!(repr(src), "['a', 'b']");
    // Plain assignment writes through, and must fit the declared type.
    assert_eq!(
        run("x: rwlock[int] = 1\ndef f():\n    x = 5\nf()\nx"),
        Value::I64(5)
    );
    assert_eq!(fails("x: rwlock[int] = 1\nx = 'a'"), ErrorClass::Type);
}

#[test]
fn rlock_is_read_only_in_nested_functions() {
    assert_eq!(
        run("cfg: rlock[int] = 3\ndef f(): return cfg * 2\ncfg = 4\nf()"),
        Value::I64(8)
    );
    // The error happens when the function is declared, not called.
    let e = Interpreter::new()
        .run("x: rlock[int] = 1\ndef f():\n    x += 1")
        .unwrap_err();
    assert_eq!(e.class(), ErrorClass::Permission);
    assert!(e.to_string().contains("may only be read"), "{e}");
    assert_eq!(
        fails("x: rlock[list[int]] = []\ndef f():\n    x.append(1)"),
        ErrorClass::Permission
    );
    // Shadowing with an annotation is fine.
    assert_eq!(
        run("x: rlock[int] = 1\ndef f():\n    x: int = 5\n    x += 1\n    return x\nf()"),
        Value::I64(6)
    );
}

#[test]
fn wlock_is_write_only_in_nested_functions() {
    let src = "log: wlock[list[str]] = []
def note(s: str):
    log.append(s)
note('hi')
log";
    assert_eq!(repr(src), "['hi']");
    assert_eq!(
        fails("x: wlock[int] = 1\ndef f(): return x"),
        ErrorClass::Permission
    );
    assert_eq!(
        fails("x: wlock[int] = 1\ndef f():\n    x += 1"),
        ErrorClass::Permission
    );
    assert_eq!(
        fails("x: wlock[list[int]] = [1]\ndef f():\n    x.pop()"),
        ErrorClass::Permission
    );
}

#[test]
fn lock_permissions_propagate_through_nesting_and_lambdas() {
    assert_eq!(
        fails("x: rlock[int] = 1\ndef f():\n    def g():\n        x += 1"),
        ErrorClass::Permission
    );
    assert_eq!(
        run("x: rwlock[int] = 1\ndef f():\n    def g():\n        x += 1\n    g()\nf()\nx"),
        Value::I64(2)
    );
    assert_eq!(
        fails("x: wlock[int] = 1\nf = lambda: x"),
        ErrorClass::Permission
    );
    assert_eq!(fails("def f(x: rwlock[int]): pass"), ErrorClass::Syntax);
    assert_eq!(fails("x: list[rwlock[int]] = []"), ErrorClass::Syntax);
}

#[test]
fn lock_reference_cycles_are_rejected() {
    // The mutual-recursion trick would close a cycle: fns -> is_even -> fns.
    let src = "fns: rwlock[dict[str, Callable]] = {}
def is_even(n: int) -> bool:
    return True if n == 0 else fns['odd'](n - 1)
fns['even'] = is_even";
    assert_eq!(fails(src), ErrorClass::Value);
    // Every way of writing a lock is checked.
    let f = "def f(): return x\n";
    assert_eq!(
        fails(&format!("x: rwlock[Any] = 0\n{f}x = f")),
        ErrorClass::Value
    );
    assert_eq!(
        fails(&format!("x: rwlock[list[Callable]] = []\n{f}x.append(f)")),
        ErrorClass::Value
    );
    assert_eq!(
        fails(&format!("x: rwlock[Any] = 0\n{f}x = [(1, map(f, []))]")),
        ErrorClass::Value
    );
    // Indirectly, through another lock.
    let src = "a: rwlock[Any] = 0
b: rwlock[Any] = 0
def fa(): return a
b = fa
def fb(): return b
a = fb";
    assert_eq!(fails(src), ErrorClass::Value);
    // The lock is unchanged after the rejected write.
    let mut i = Interpreter::new();
    assert!(i.run(&format!("x: rwlock[Any] = 0\n{f}x = f")).is_err());
    assert_eq!(i.get("x"), Some(Value::I64(0)));
    // Functions that don't capture the lock are fine.
    assert_eq!(
        run("fs: rwlock[list[Callable]] = []\ndef g(): return 1\nfs.append(g)\nfs[0]()"),
        Value::I64(1)
    );
}

#[test]
fn annotations_shadow_for_the_rest_of_the_block() {
    // Plain variables.
    assert_eq!(
        run("x = 1\nif True:\n    x: str = 'a'\n    x = 'b'\n    x += 'c'\nx"),
        Value::I64(1)
    );
    assert_eq!(
        repr("x = 1\nout = []\nfor i in range(2):\n    x: str = 'a'\n    out.append(x)\n(x, out)"),
        "(1, ['a', 'a'])"
    );
    // Locks: a shadow in a sub-block hides the lock until the block ends.
    assert_eq!(
        run("c: rwlock[int] = 0\nif True:\n    c: int = 5\n    c += 1\n    c = 'now a str'\nc"),
        Value::I64(0)
    );
    assert_eq!(
        run("c: rwlock[int] = 0\nif True:\n    c += 1\nc"),
        Value::I64(1)
    );
    // A shadowing lock in a sub-block is its own lock.
    assert_eq!(
        run(
            "c: rwlock[int] = 0\nif True:\n    c: rwlock[int] = 10\n    def f():\n        c += 1\n    f()\nc"
        ),
        Value::I64(0)
    );
    // Inside a function, the same rules apply to captured variables.
    assert_eq!(
        run(
            "x: rlock[int] = 1\ndef f():\n    if True:\n        x: int = 2\n        x += 1\n    return x\nf()"
        ),
        Value::I64(1)
    );
}

#[test]
fn host_shared_locks() {
    let mut i = Interpreter::new();
    let total = i
        .share("total", LockKind::ReadWrite, Type::I64, 0i64)
        .unwrap();
    i.run("def add(n: int):\n    total += n\nadd(2)\nadd(3)")
        .unwrap();
    assert_eq!(total.get(), Value::I64(5));
    total.set(100i64).unwrap();
    assert_eq!(i.eval("total").unwrap(), Value::I64(100));
    assert!(total.set("x").is_err());
    // The host can't close a cycle either.
    let f = i.share("f", LockKind::ReadWrite, Type::Any, 0i64).unwrap();
    i.run("def g(): return f").unwrap();
    assert_eq!(
        f.set(i.get("g").unwrap()).unwrap_err().class(),
        ErrorClass::Value
    );
}

// --------------------------------------------------------------- functions

#[test]
fn call_time_type_checks() {
    assert_eq!(fails("def f(a: u8): pass\nf('x')"), ErrorClass::Type);
    assert_eq!(fails("def f() -> int: return 'x'\nf()"), ErrorClass::Type);
    assert_eq!(fails("def f() -> int: pass\nf()"), ErrorClass::Type);
    assert_eq!(fails("def f(a): pass\nf()"), ErrorClass::Type);
    assert_eq!(fails("x = 1\nx()"), ErrorClass::Type);
    assert_eq!(run("def id(a): return a\nid('s')"), Value::from("s"));
}

#[test]
fn callable_types() {
    let src = "def apply(f: Callable[[int], int], x: int) -> int: return f(x)
def inc(n: int) -> int: return n + 1
apply(inc, 1)";
    assert_eq!(run(src), Value::I64(2));
    // Unannotated lambdas are compatible with any signature of their arity.
    assert_eq!(
        run(
            "def apply(f: Callable[[int], int], x: int) -> int: return f(x)\napply(lambda n: n * 2, 4)"
        ),
        Value::I64(8)
    );
    assert_eq!(
        fails("def apply(f: Callable[[int], int]): pass\ndef g(s: str) -> int: return 1\napply(g)"),
        ErrorClass::Type
    );
    assert_eq!(
        fails("def apply(f: Callable[[int], int]): pass\napply(lambda a, b: a)"),
        ErrorClass::Type
    );
    assert_eq!(
        run("def apply(f: Callable): return f()\napply(lambda: 3)"),
        Value::I64(3)
    );
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
    assert_eq!(fails(src), ErrorClass::Limit);
    assert_eq!(fails("(lambda f: f(f))(lambda f: f(f))"), ErrorClass::Limit);
}

// ---------------------------------------------------------------- builtins

#[test]
fn print_builtin() {
    assert_eq!(
        output("print(1, 'a', [b'x'], 2.0, None)\nprint()"),
        "1 a [b'x'] 2.0 None\n\n"
    );
    assert_eq!(fails("print(print)"), ErrorClass::Type);
    assert_eq!(run("print()"), Value::None);
}

#[test]
fn map_is_lazy_and_reiterable() {
    assert_eq!(repr("map(lambda x: x, [1])"), "<map object>");
    assert_eq!(repr("list(map(lambda x: x * x, [1, 2, 3]))"), "[1, 4, 9]");
    assert_eq!(repr("list(map(str, {1: 'a'}))"), "['1']");
    // Nothing runs until iterated.
    assert_eq!(output("m = map(print, [1, 2])"), "");
    assert_eq!(
        output("m = map(print, [1, 2])\nfor _ in m:\n    pass"),
        "1\n2\n"
    );
    // Views snapshot their source and can be iterated again.
    assert_eq!(
        repr("xs = [1, 2]\nm = map(lambda x: x * 10, xs)\nxs.append(3)\n[list(m), list(m)]"),
        "[[10, 20], [10, 20]]"
    );
    assert_eq!(run("3 in map(lambda x: x + 1, [1, 2])"), Value::Bool(true));
    assert_eq!(
        run("def sq(x: int) -> int: return x * x\nr: list[int] = list(map(sq, []))\nr == []"),
        Value::Bool(true)
    );
    assert_eq!(fails("map(1, [1])"), ErrorClass::Type);
    assert_eq!(fails("len(map(str, [1]))"), ErrorClass::Type);
    assert_eq!(
        repr("list(map(lambda x: x + 1, map(lambda x: x * 2, range(3))))"),
        "[1, 3, 5]"
    );
}

#[test]
fn range_builtin() {
    assert_eq!(repr("range(3)"), "range(0, 3)");
    assert_eq!(run("len(range(0, 10, 3))"), Value::I64(4));
    assert_eq!(run("len(range(10, 0, -3))"), Value::I64(4));
    assert_eq!(run("4 in range(0, 10, 2)"), Value::Bool(true));
    assert_eq!(run("5 in range(0, 10, 2)"), Value::Bool(false));
    assert_eq!(fails("range(0, 10, 0)"), ErrorClass::Value);
    assert_eq!(fails("x: u8 = 1\ny: i8 = 2\nrange(x, y)"), ErrorClass::Type);
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("femtopython-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn with_opener(opener: Opener) -> Interpreter {
    Interpreter::with_config(Config {
        opener,
        ..Config::default()
    })
}

#[test]
fn paths_and_open() {
    assert_eq!(
        repr("Path('a') / 'b'"),
        format!("Path('a{}b')", std::path::MAIN_SEPARATOR)
    );
    // Denied by default.
    assert_eq!(fails("open('/etc/hostname')"), ErrorClass::Permission);

    let dir = TempDir::new("open");
    std::fs::write(dir.0.join("f.txt"), "contents").unwrap();
    let mut i = with_opener(Opener::filesystem());
    i.set("dir", dir.0.clone());
    assert_eq!(
        i.run("open(dir / 'f.txt').read()").unwrap(),
        Value::from("contents")
    );
    assert_eq!(
        i.run("open(dir / 'missing').read()").unwrap_err().class(),
        ErrorClass::Io
    );
    // `.read()` gives bytes when the context wants bytes, without decoding.
    std::fs::write(dir.0.join("bin"), [0xff, 0x00]).unwrap();
    assert_eq!(
        i.run("b: bytes = open(dir / 'bin').read()\nb").unwrap(),
        Value::Bytes([0xff, 0x00].as_slice().into())
    );
    assert_eq!(
        i.run("open(dir / 'bin').read()").unwrap_err().class(),
        ErrorClass::Value
    );
    assert_eq!(
        i.run("def size(b: bytes) -> int: return len(b)\nsize(open(dir / 'bin').read())")
            .unwrap(),
        Value::I64(2)
    );
}

#[test]
fn allowlisted_directories_confine_open() {
    let root = TempDir::new("allow-dir");
    let data = root.0.join("data");
    std::fs::create_dir_all(data.join("sub")).unwrap();
    std::fs::write(data.join("a.txt"), "a").unwrap();
    std::fs::write(data.join("sub").join("b.txt"), "b").unwrap();
    std::fs::write(root.0.join("secret.txt"), "secret").unwrap();

    let mut i = with_opener(Opener::allowlist(Allowlist::new().dir(&data).unwrap()));
    i.set("root", root.0.clone());
    i.set("data", data.clone());
    assert_eq!(
        i.run("open(data / 'a.txt').read()").unwrap(),
        Value::from("a")
    );
    assert_eq!(
        i.run("open(data / 'sub' / 'b.txt').read()").unwrap(),
        Value::from("b")
    );
    assert_eq!(
        i.run("open(data / 'sub' / '..' / 'a.txt').read()").unwrap(),
        Value::from("a")
    );
    // Outside the directory, directly or via `..`.
    for escape in [
        "open(root / 'secret.txt')",
        "open(data / '..' / 'secret.txt')",
        "open(data / 'sub' / '..' / '..' / 'secret.txt')",
        "open('/etc/hostname')",
    ] {
        let e = i.run(escape).unwrap_err();
        assert!(
            matches!(e.class(), ErrorClass::Permission | ErrorClass::Io),
            "{escape}: {e}"
        );
        assert!(!e.to_string().contains("secret\""), "{e}");
    }
    #[cfg(unix)]
    {
        // A symlink inside the directory pointing out of it doesn't escape.
        std::os::unix::fs::symlink(root.0.join("secret.txt"), data.join("link")).unwrap();
        assert!(i.run("open(data / 'link').read()").is_err());
        std::os::unix::fs::symlink(&root.0, data.join("up")).unwrap();
        assert!(i.run("open(data / 'up' / 'secret.txt').read()").is_err());
    }
}

#[test]
fn allowlisted_files_allow_exactly_that_file() {
    let root = TempDir::new("allow-file");
    std::fs::write(root.0.join("ok.json"), "[1]").unwrap();
    std::fs::write(root.0.join("other.json"), "[2]").unwrap();
    let ok = root.0.join("ok.json");
    let mut i = with_opener(Opener::allowlist(Allowlist::new().file(&ok).unwrap()));
    i.set("root", root.0.clone());
    assert_eq!(
        i.run("json.loads(open(root / 'ok.json').read())")
            .unwrap()
            .repr(),
        "[1]"
    );
    assert_eq!(
        i.run("open(root / 'other.json')").unwrap_err().class(),
        ErrorClass::Permission
    );
    // Bad allowlist entries are reported to the host up front.
    assert!(Allowlist::new().dir(root.0.join("nope")).is_err());
}

#[test]
fn custom_openers_are_invisible_to_scripts() {
    use std::collections::HashMap;
    let files: HashMap<&str, &str> = [("config.json", r#"{"debug": true}"#)].into();
    let opener = Opener::custom(move |p| {
        let name = p.to_str().unwrap_or_default();
        match files.get(name) {
            Some(text) => Ok(Box::new(std::io::Cursor::new(text.as_bytes().to_vec())) as Reader),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no such file",
            )),
        }
    });
    let mut i = with_opener(opener);
    assert_eq!(
        i.run("cfg = json.loads(open('config.json').read())\ncfg['debug']")
            .unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        i.run("open(Path('config.json')).read()").unwrap(),
        Value::from(r#"{"debug": true}"#)
    );
    let e = i.run("open('nope')").unwrap_err();
    assert!(
        matches!(&e.kind, ErrorKind::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound)
    );
}

#[test]
fn methods_must_be_called() {
    assert_eq!(fails("{1: 2}.keys"), ErrorClass::Type);
    assert_eq!(fails("x = 1\nx.append(2)"), ErrorClass::Type);
}

// --------------------------------------------------------- syntax & limits

#[test]
fn unsupported_syntax_is_rejected_clearly() {
    let e = Interpreter::new().run("import os").unwrap_err();
    assert_eq!(e.class(), ErrorClass::Syntax);
    assert!(e.to_string().contains("not supported"), "{e}");
    assert_eq!(fails("x = 1, 2"), ErrorClass::Syntax);
    assert_eq!(fails("x: nosuchtype = 1"), ErrorClass::Syntax);
    assert_eq!(fails("x: set[float] = {}"), ErrorClass::Syntax);
    assert_eq!(
        fails("while True:\n    pass\nelse:\n    pass"),
        ErrorClass::Syntax
    );
}

#[test]
fn deeply_nested_input_is_rejected() {
    // Redundant parentheses don't nest the AST, so they are harmless.
    let parens = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
    assert_eq!(run(&parens), Value::I64(1));
    let lists = format!("{}1{}", "[".repeat(10_000), "]".repeat(10_000));
    assert_eq!(fails(&lists), ErrorClass::Limit);
    let negs = format!("{}1", "-".repeat(10_000));
    assert_eq!(fails(&negs), ErrorClass::Limit);
    let long_chain = vec!["1"; 50_000].join(" + ");
    assert_eq!(fails(&long_chain), ErrorClass::Limit);
    let deep_type = format!("x: {}int{} = 1", "list[".repeat(100), "]".repeat(100));
    assert_eq!(fails(&deep_type), ErrorClass::Syntax);
}

#[test]
fn deeply_nested_values_are_rejected() {
    assert_eq!(
        fails("a: Any = 1\nfor i in range(100):\n    a = [a]"),
        ErrorClass::Limit
    );
    assert_eq!(
        fails("m: Any = map(str, [1])\nfor i in range(100):\n    m = map(str, m)"),
        ErrorClass::Limit
    );
}

#[test]
fn string_growth_is_bounded() {
    let mut i = Interpreter::with_config(Config {
        max_str_len: 1024,
        ..Config::default()
    });
    let src = "s = 'xx'\nfor i in range(20):\n    s = s + s";
    assert_eq!(i.run(src).unwrap_err().class(), ErrorClass::Limit);
}

#[test]
fn input_status_for_repl() {
    assert_eq!(input_status("1 +"), InputStatus::Invalid);
    assert_eq!(input_status("(1 +"), InputStatus::Incomplete);
    assert_eq!(input_status("def f():"), InputStatus::Incomplete);
    assert_eq!(input_status("if x:"), InputStatus::Incomplete);
    assert_eq!(
        input_status("for i in x:\n    if i:\n"),
        InputStatus::Incomplete
    );
    assert_eq!(
        input_status("for i in x:\n    if i:\n        pass\n"),
        InputStatus::Complete
    );
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
    i.register_raw_fn("double", |args| match args {
        [Value::I64(n)] => Ok(Value::I64(n * 2)),
        _ => Err(femtopython::Error::new(ErrorKind::Type(
            "double(int)".into(),
        ))),
    });
    i.run("def add(a: int, b: int) -> int: return a + b\ny = double(x) + 2")
        .unwrap();
    assert_eq!(i.get("y"), Some(Value::I64(42)));
    let add = i.get("add").unwrap();
    assert_eq!(
        i.call(&add, vec![Value::I64(1), Value::I64(2)]).unwrap(),
        Value::I64(3)
    );
    assert_eq!(i.eval("x * 2").unwrap(), Value::I64(40));
    // State persists across runs, and errors keep earlier effects.
    assert!(i.run("z = 1\nundefined_name").is_err());
    assert_eq!(i.get("z"), Some(Value::I64(1)));
}

#[test]
fn interpreter_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Interpreter>();
    let mut i = Interpreter::new();
    i.run("x = 41").unwrap();
    let v = std::thread::spawn(move || i.eval("x + 1").unwrap())
        .join()
        .unwrap();
    assert_eq!(v, Value::I64(42));
}

#[test]
fn errors_are_structured() {
    use femtopython::LimitExceeded;
    let mut i = Interpreter::with_config(Config {
        max_steps: Some(100),
        ..Config::default()
    });
    let e = i.run("while True:\n    pass").unwrap_err();
    assert!(
        matches!(e.kind, ErrorKind::Limit(LimitExceeded::Steps(100))),
        "{e:?}"
    );
    let e = Interpreter::new().run("nope + 1").unwrap_err();
    assert!(
        matches!(&e.kind, ErrorKind::Name(n) if n == "nope"),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "NameError: name 'nope' is not defined");
    let e = Interpreter::new().run("1 // 0").unwrap_err();
    assert!(matches!(e.kind, ErrorKind::ZeroDivision));
    assert_eq!(e.class().name(), "ZeroDivisionError");
    let e = Interpreter::new()
        .run("def f(n): return f(n)\nf(1)")
        .unwrap_err();
    assert!(matches!(
        e.kind,
        ErrorKind::Limit(LimitExceeded::EvalDepth(_))
    ));
    // Errors work with `?` and `Box<dyn Error>`.
    fn host() -> Result<(), Box<dyn std::error::Error>> {
        Interpreter::new().run("1 + 'a'")?;
        Ok(())
    }
    assert!(host().unwrap_err().to_string().starts_with("TypeError"));
}

#[test]
fn range_inference_is_int_only() {
    assert_eq!(repr("x: u8 = 1\nlist(range(x, 3))"), "[1, 2]");
    assert_eq!(run("r: Iterable[u8] = range(3)\nlist(r)[0]"), Value::U8(0));
    assert_eq!(
        run("def f(r: Iterable[i8]) -> int: return len(r)\nf(range(-3, 3))"),
        Value::I64(6)
    );
    assert_eq!(fails("range(1.5)"), ErrorClass::Type);
    assert_eq!(fails("range(0, 2.5)"), ErrorClass::Type);
    assert_eq!(fails("range('3')"), ErrorClass::Type);
    assert_eq!(fails("x: u8 = 1\ny = 5\nrange(x, y)"), ErrorClass::Type);
    assert_eq!(fails("x: u8 = 1\nrange(x, 300)"), ErrorClass::Overflow);
}

use femtopython::{ErrorClass, Interpreter, Value};

fn run(src: &str) -> Value {
    Interpreter::new()
        .run(src)
        .unwrap_or_else(|e| panic!("{}", e.render(src)))
}

fn fails(src: &str) -> ErrorClass {
    match Interpreter::new().run(src) {
        Ok(v) => panic!("expected {src:?} to fail, got {v:?}"),
        Err(e) => e.class(),
    }
}

#[test]
fn loads_scalars() {
    assert_eq!(run("json.loads('1')"), Value::I64(1));
    assert_eq!(run("json.loads('-0')"), Value::I64(0));
    assert_eq!(run("json.loads('1.5')"), Value::F64(1.5));
    assert_eq!(run("json.loads('1e2')"), Value::F64(100.0));
    assert_eq!(run("json.loads('true')"), Value::Bool(true));
    assert_eq!(run("json.loads('null')"), Value::None);
    assert_eq!(run("json.loads(' \"hi\" ')"), Value::from("hi"));
    assert_eq!(
        run(r#"json.loads('"a\\n\\u00e9\\ud83d\\ude00\\/"')"#),
        Value::from("a\né😀/")
    );
}

#[test]
fn loads_documents() {
    let src = r#"doc = json.loads('{"users": [{"id": 1, "name": "a"}, {"id": 2, "name": null}], "ok": true}')
doc['users'][1]['id'] + 40"#;
    assert_eq!(run(src), Value::I64(42));
    assert_eq!(
        run("json.loads('[1, \"a\", null]')").repr(),
        "[1, 'a', None]"
    );
    assert_eq!(run("json.loads('[1, 2]')").repr(), "[1, 2]");
    assert_eq!(run("json.loads('[]')").repr(), "[]");
    assert_eq!(run("json.loads('{}')").repr(), "{}");
    // Duplicate keys: the last value wins, as in Python.
    assert_eq!(run("json.loads('{\"a\": 1, \"a\": 2}')").repr(), "{'a': 2}");
    // Loaded dicts are flexible (JSON objects), so they accept new types.
    assert_eq!(
        run("d = json.loads('{\"a\": 1}')\nd['b'] = 'x'\nd['b']"),
        Value::from("x")
    );
    // And can be checked against annotations.
    assert_eq!(
        run("d: dict[str, int] = json.loads('{\"a\": 1}')\nd['a']"),
        Value::I64(1)
    );
}

#[test]
fn loads_bytes() {
    assert_eq!(run("json.loads(b'[1, 2]')").repr(), "[1, 2]");
    assert_eq!(run("json.loads(b'\"\\xc3\\xa9\"')"), Value::from("é"));
    assert_eq!(fails("json.loads(b'\"\\xff\"')"), ErrorClass::Value);
    assert_eq!(fails("json.loads(1)"), ErrorClass::Type);
}

#[test]
fn loads_rejects_invalid_json() {
    for bad in [
        "",
        "01",
        "1.",
        ".5",
        "+1",
        "[1,]",
        "{\"a\":1,}",
        "{a: 1}",
        "'a'",
        "[1 2]",
        "nul",
        "\"\\x\"",
        "\"\\ud800\"",
        "NaN",
        "1 2",
        "\"a",
        "{\"a\"}",
        "\"\t\"",
    ] {
        let src = format!("json.loads({bad:?})");
        assert_eq!(fails(&src), ErrorClass::Value, "{bad:?} should be rejected");
    }
    // Integers too large for int.
    assert_eq!(
        fails("json.loads('99999999999999999999')"),
        ErrorClass::Value
    );
}

#[test]
fn loads_bounds_nesting() {
    let deep = format!("json.loads('{}{}')", "[".repeat(10_000), "]".repeat(10_000));
    assert_eq!(fails(&deep), ErrorClass::Limit);
}

#[test]
fn dumps_matches_python_formatting() {
    assert_eq!(
        run("json.dumps({'a': 1, 'b': [True, None], 'c': 2.5, 'd': 'é\"\\n'})"),
        Value::from(r#"{"a": 1, "b": [true, null], "c": 2.5, "d": "\u00e9\"\n"}"#)
    );
    assert_eq!(run("json.dumps((1, 2))"), Value::from("[1, 2]"));
    assert_eq!(run("json.dumps([])"), Value::from("[]"));
    assert_eq!(run("json.dumps('😀')"), Value::from(r#""\ud83d\ude00""#));
    assert_eq!(run("x: u8 = 7\njson.dumps(x)"), Value::from("7"));
    assert_eq!(run("json.dumps(1.0)"), Value::from("1.0"));
    // Non-str keys are stringified, like Python.
    assert_eq!(
        run("json.dumps({1: 'a', 2: 'b'})"),
        Value::from(r#"{"1": "a", "2": "b"}"#)
    );
    assert_eq!(run("json.dumps({True: 1})"), Value::from(r#"{"true": 1}"#));
}

#[test]
fn dumps_rejects_non_json_values() {
    assert_eq!(fails("json.dumps({1, 2})"), ErrorClass::Type);
    assert_eq!(fails("json.dumps(b'x')"), ErrorClass::Type);
    assert_eq!(fails("json.dumps(print)"), ErrorClass::Type);
    assert_eq!(fails("json.dumps(map(str, [1]))"), ErrorClass::Type);
    assert_eq!(fails("x = 1.0 / 0.0\njson.dumps(x)"), ErrorClass::Value);
    assert_eq!(fails("json.dumps({(1, 2): 1})"), ErrorClass::Type);
}

#[test]
fn dumps_infers_bytes_from_context() {
    assert_eq!(
        run("b: bytes = json.dumps([1])\nb"),
        Value::Bytes(b"[1]".as_slice().into())
    );
    assert_eq!(
        run("def send(data: bytes) -> int: return len(data)\nsend(json.dumps({'a': 1}))"),
        Value::I64(8)
    );
    assert_eq!(
        run("def f() -> bytes: return json.dumps('é')\nf()"),
        Value::Bytes(br#""\u00e9""#.as_slice().into())
    );
    // Without context it's a str.
    assert_eq!(run("s = json.dumps(1)\ns"), Value::from("1"));
}

#[test]
fn round_trips() {
    let src = r#"doc = {'name': 'x', 'n': [1, 2, 3], 'nested': {'ok': True, 'v': None}, 'f': 0.1}
json.loads(json.dumps(doc)) == doc"#;
    assert_eq!(run(src), Value::Bool(true));
    assert_eq!(
        run("json.loads(json.dumps(b'') if False else b'[1]')").repr(),
        "[1]"
    );
}

#[test]
fn json_functions_are_first_class() {
    assert_eq!(
        run("list(map(json.loads, ['1', '[2]']))").repr(),
        "[1, [2]]"
    );
    assert_eq!(run("f = json.dumps\nf(None)"), Value::from("null"));
    assert_eq!(run("json").repr(), "<module 'json'>");
    assert_eq!(fails("json.nope('1')"), ErrorClass::Type);
    assert_eq!(fails("json(1)"), ErrorClass::Type);
}

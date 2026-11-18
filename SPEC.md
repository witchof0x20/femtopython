This project is a small implementation of a subset of Python. The goal is embedding in Rust programs, simple tasks like acting as a calculator, and environments where embedding a fully compliant Python interpreter risks security issues. I also want it to be more typed than normal python, closer to rust.

This interpreter should support the following features, and no more.

## Types
### Rust-like primitive types
Primitives should be usable like in python, where they can act as functions. If a `From` exists in rust the cast always succeeds and returns `T`. If it only exists as a `TryFrom` (or `FromStr` for parsing a str), the cast returns `Optional[T]`, which is `None` when it fails. If neither exists, it's a type error. There is no `as`.
* `u8(x)` where x is an i64: `TryFrom`, so `Optional[u8]`. `u8(300)` is `None`
* `int(x)` where x is a u8: `From`, so always an int
* `int("42")` is 42, `int("nope")` is `None`
* `float(x)` where x is an i64: error, no `From<i64> for f64` (lossy)
* `int(2.5)`: error, no float to int conversion at all
* `str(x)` works on anything that isn't a function (rust's `Display`). `str(bytes)` and `str(Path)` are fallible (utf-8), so `Optional[str]`
* `bytes(str)`, `bytes(list[u8])`, `Path(str)` always succeed
#### Integers
* i8,i16,i32,i64,u8,u16,u32,u64
* int is alias to i64
#### Floats
* f32,f64
* float is alias to f64
#### Bool
* bool, `True` / `False`
* No truthiness. `if`, `while`, `and`, `or`, `not` and conditional expressions all need an actual bool
#### None / Optional
* `None` is the only value of type None
* `Optional[T]` is `T` or `None`. `Optional[Optional[T]]` is just `Optional[T]`
* Check it with `x is None` / `x is not None`. `is` only exists for this, the right side has to be a literal `None`
* `==` / `!=` against `None` work on anything (always false / true unless both are None)
#### Any
* `Any` accepts any value. Never inferred, except for dict values (see Dicts). Exists so things like json can be represented
#### Strings
* str (roughly, rust String. indexing and iterating it should be in terms of chars)
* bytes (roughly, rust &[u8], indexing and iterating should return u8)
* There's no char type, indexing/iterating a str gives a 1-char str
* Negative indexes count from the end, like python
* `+` concatenates str+str and bytes+bytes
* `in` is substring for str, and byte-or-subsequence for bytes
#### Lists
* `list[T]`
* No mixed types. `[1, None]` is fine though, it's `list[Optional[int]]`
* Methods: `.append(x)` `.extend(iterable)` `.pop()` `.pop(i)` `.clear()`
### Dicts
* `dict[K, V]`
* No mixed key types. Keys have to be hashable (no floats, same as rust)
* Values can be mixed, think json. Literal values in a dict are typed on their own, so `{'a': 1, 'b': 2.5}` keeps the 1 an int. A dict built from literals widens its value type to `Any` when it sees a new type. Once a dict has been checked against an annotation (`d: dict[str, int]`, a typed param, a typed return) its value type is fixed and mismatches are an error
* Insertion ordered
* `.keys() .values() .items()` return lazy views (see Iterables)
* `.get(k)` returns `Optional[V]`, `.get(k, default)`, `.pop(k)`, `.pop(k, default)`, `.clear()`
* Iterating a dict iterates its keys
### Sets
* `set[T]`
* No mixed types, elements have to be hashable
* `{}` is an empty dict, an empty set is `set()`
* Operators: `|` union, `&` intersection, `-` difference, `^` symmetric difference, `<= < >= >` subset/superset
* Methods: `.add(x)` `.remove(x)` (error if missing) `.discard(x)` `.clear()`
* Insertion ordered
### Tuples
* Basically same as rust, but mixed-type tuples can't be cast or iterated.
* `(1,)` is a one-tuple, `()` is empty
* Immutable
### Iterables
* `Iterable[T]`, a lazy view. `map`, `range`, `.keys()`, `.values()`, `.items()` return these
* Nothing runs until you iterate. `map(print, xs)` prints nothing until it's looped over
* Views snapshot their source when created and can be iterated more than once
* Anything iterable works in `for`, `in`, `list()`, `set()`, `map()` and `.extend()`: lists, sets, dicts (keys), strs (chars), bytes (u8), single-type tuples, and views
* `len()` works on views where the length is known without running code (not `map`)
### Functions
* `Callable` is any function. `Callable[[A, B], R]` is a function taking A and B and returning R, python style
* Unannotated params / returns are unknown and are compatible with anything
### Paths / files
* `Path`, basically a PathBuf. `/` joins (`Path("a") / "b"`)
* `File`, what `open` returns. `.read()` reads the rest of it as a str, or as bytes if the context wants bytes (`b: bytes = f.read()`, a bytes param, a `-> bytes` return), in which case it isn't utf-8 checked
### Copy on write
Lists, dicts and sets are values, like owned rust collections. `b = a` then `b[0] = 1` does not change `a`. Under the hood they're shared and only copied when written to, so passing them around is cheap.

## Variables / literals
Declare normally: `a = 2` (infer int, float, str, bytes from the const value)
Declare with type: `a: int = 2`
b"" byte strings, "" strings, both with `'` or `"`, single line. Escapes: `\\ \' \" \n \r \t \0 \xHH`, plus `\uXXXX \UXXXXXXXX` in str. bytes literals are ascii only.
Integer literals: decimal, `0x`, `0o`, `0b`, `_` separators. Floats: `1.5`, `1e3`, `.5`
Redeclaration is okay (like Rust's let) but it's a redeclaration, not an assignment.

### Literals take their type from context
Like rust's integer literals. A literal number becomes whatever type the context needs, checked for range:
* `x: u8 = 5`, then `x + 1`, `x += 1`, `x * 2` all stay u8
* function args take the param's type, `f(7)` where `f(a: u16)`
* `xs: list[u8] = [1, 2]`, `xs.append(3)`
* two bare literals unify, preferring float: `2 * 3.5` is `7.0`, so it works as a calculator
* an int literal can become a float if it's exactly representable (`x: float = 2`)
* `-128` fits i8, it's one literal
* with no context: int / float. A literal that doesn't fit is an error (`x: u8 = 300`)

### Scopes
* Every function call, and every `if`/`elif`/`else`/`while`/`for` body, opens a new scope for declarations. Something declared in a block is gone after the block
* `x: T = v` always declares a new `x` in the current scope (shadowing anything outside)
* `x = v` where `x` is in the current scope: redeclaration. If it has an annotated type and `v` fits it, the annotation is kept. If not, it's a new variable with a new type
* `x = v` where `x` is in an enclosing block of the same function: writes to it, no special treatment. The type has to stay compatible (otherwise annotate to shadow)
* `x = v` where `x` doesn't exist: declares it
* Same rule for locks and for plain variables: an annotation shadows for the rest of the scope it's in, including sub-blocks like an `if` body. Inside the shadow, `x = v` and `x += v` hit the shadow. Once the block ends the outer `x` (or lock) is back, untouched
* for loop variables live in the loop body
* Variables keep their annotated type, so `best: Optional[int] = None` then `best = x` in a loop works

### Locks
`rwlock[T]`, `rlock[T]`, `wlock[T]` are how state gets shared with functions. Only allowed as the outermost annotation on a declaration (not params, returns, or inside other types).
* Reads and writes are transparent. `count += 1`, `count = 5`, `log.append(x)` all write the shared value. Writes must fit `T`
* In the scope that declares it, all three are read-write
* Functions / lambdas declared inside get:
  * `rwlock[T]`: read-write (the rust equivalent)
  * `rlock[T]`: read only. Writing it is an error at declaration time
  * `wlock[T]`: write only. Reading it is an error at declaration time (`.append()` etc is fine, `.pop()` and `+=` read)
* "Declaration time" is when the `def`/`lambda` runs, before it's ever called. Nested functions keep the narrowest access
* To get a local that shadows a lock, annotate it: `x: int = ...`
* A lock can't end up holding a function that captures that same lock, directly or through other locks (e.g. `fns: rwlock[list[Callable]]` and appending a function that uses `fns`). That write is an error and the lock is left unchanged. Every other value is immutable or copy on write, so this is the only way to make a reference cycle, and this means everything always gets freed
* The host can create these too and read/write them from rust while a script runs

## Functions
Still first class, as before.
Type definitions not required, but calls are typechecked at call time (args against param types, return against return type).
Declarations inside functions shadow.
All variables outside a function are copied once, when the function is declared, and only the ones it actually uses. After that it never sees changes from outside, and writing to them only changes the function's copy. Locks are the exception, see above.
A function can always call itself (recursion). Names declared after a function aren't visible to it, so there's no mutual recursion between functions.
`lambda a, b: expr`, same capture rules, no annotations.
No default args, no keyword args, no varargs.

## Control flow
* `if` / `elif` / `else`
* `while`
* `for x in iterable`, and `for k, v in iterable` to unpack tuples
* `break`, `continue`, `pass`, `return`
* `a if cond else b`
* No `else` on loops, no `try`, no `with`, no comprehensions

## Operations
All operations should operate basically the same as their equivalent rust primitives
* Both sides have to be the same type. No implicit int/float mixing, no mixing u8 and u16. Convert explicitly
* Integer overflow is an error (like a rust debug build), as is dividing by zero
* Arithmetic `+ - * / // % **`
  * `/` on ints truncates like rust, `%` is rust's remainder (sign follows the left side)
  * `//` is floor division like python
  * `**` binds tighter than unary minus and is right associative like python. For ints the exponent can be any int type, but can't be negative
  * shifts also take any int type on the right
  * unary `-` on unsigned is an error
* Comparators `> >= == != <= <`
  * don't chain (`a < b < c` is a syntax error), like rust
  * lists and tuples compare lexicographically, sets by subset
* Membership `in`, `not in`
* `is None`, `is not None`
* Logical `and or not`, bools only, short circuiting
* Bitwise `& | ^ ~ << >>`, ints, and `& | ^` on bools
* Assignment `= += -= *= /= //= %= **= &= |= ^= <<= >>=`, on variables and on `x[i]` / `x[i][j]`

# Builtins outside of function types
* print(varargs of anything that aren't a function) returns None always, but is typed `Optional[bool]`. varargs are not normally allowed, this function gets special treatment
* map(f: T -> U, Iterator<Item=T>), returns a lazy `Iterable[U]`
* range(stop), range(start, stop), range(start, stop, step), lazy. Ints only. Args are all the same int type, and so are the elements. Literal args take their type from a non-literal arg (`range(x, 10)` with `x: u8` is u8), else from context (`r: Iterable[u8] = range(10)`), else int
* len(x) for str (in chars), bytes, list, dict, set, tuple, and views with a known length
* list(iterable), set(iterable), or empty with no args
* open(filename: str|Path), read only. What it actually does is completely up to the host (see Embedding), the script can't tell the difference. Off unless the host turns it on
* Path(str) (under the hood this should just be PathBuf basically and only really exist to be used for open and support / sugar)
* the type names as conversion functions, see the top
* `json.loads(s)`, s is str or bytes (utf-8). Returns `Any`
  * objects become dicts (str keys, values widen like any literal-built dict), arrays become lists (`list[Any]` if mixed), integers become int, numbers with a fraction or exponent become float
  * strict json: no trailing commas, comments, NaN/Infinity or lone surrogates. Integers that don't fit int are an error. Duplicate keys, the last one wins
  * invalid json is an error, not `None` (since `null` is valid json)
* `json.dumps(v)`, returns str, or bytes if the context wants bytes (same as `.read()`)
  * same format as python's default: `", "` and `": "` separators, non-ascii escaped
  * lists and tuples become arrays. int/bool/None dict keys get stringified like python
  * sets, bytes, functions, paths and views are errors. NaN and inf are errors
* `json` itself isn't a module system, there's no `import`. It's just a builtin namespace, and `json.loads` / `json.dumps` are first class like anything else

# Embedding
* `Interpreter` is the object. `run(src)` runs statements and returns the last expression's value, `eval(src)` evaluates one expression, `call(f, args)` calls a script function from rust. Globals stick around between runs
* `get` / `set` globals, `share(name, kind, type, value)` makes a lock global the host can keep a handle to
* `register_fn(name, closure)` exposes a rust function. Types come from the rust signature: primitives, `String`, `Bytes`, `PathBuf`, `Vec<T>`, `HashMap<K, V>`, tuples, `Option<T>` as `Optional[T]`, `Value` for anything, and `Result<T, E>` returns become script errors. Calls get checked and literals get typed like any other function. `register_raw_fn` takes `&[Value]` unchecked
* `Interpreter` is `Send`. Native functions and the print output have to be `Send + Sync`
* `input_status(src)` says whether input is complete, incomplete (keep reading) or invalid, for building a repl

## open()
The host decides what `open` does with an `Opener` in the config. The interpreter just calls it with the path and reads whatever `Read` it gets back.
* `Opener::deny()`, the default. Always a PermissionError
* `Opener::filesystem()`, regular file opens
* `Opener::allowlist(Allowlist::new().dir(path)?.file(path)?)`, whitelisted paths only
  * a dir entry allows anything beneath it. The dir is opened when it's added to the list and every open is resolved beneath that handle (cap-std), so `..` and symlinks can't get out of it, even if the filesystem changes later
  * a file entry allows exactly that file name in its (also pinned) directory
  * anything else is a PermissionError. Bad entries (missing dirs etc) are an error for the host when building the list, not later
* `Opener::custom(f)`, anything else, e.g. serving files from memory. A `PermissionDenied` io error from it becomes a PermissionError in the script, anything else an IOError

## Errors
Everything that goes wrong comes back to the caller as an `Error`: an `ErrorKind` (thiserror enum) saying exactly what happened, plus the span. Scripts can't catch anything (no `try`), so an error always ends the run and goes to the host.
* `ErrorKind` variants: `Syntax`, `IncompleteInput`, `Type`, `Name` (holds the name), `Value`, `Index`, `Key`, `ZeroDivision`, `Overflow`, `Io` (keeps the `io::Error` as its `source()`), `Permission`, `Limit` (which limit, as a `LimitExceeded`: `ExprDepth`, `NestingDepth`, `EvalDepth`, `Steps`, `Size`), `Interrupted`, `Native`
* `Native` is an error returned by a host function, passed through untouched. `error.downcast_native::<MyError>()` gets the original back, and it's also the `source()`. Host functions get this by returning `Result<T, Box<dyn Error + Send + Sync>>` or using `Error::native(e)`. Returning a `String`/`&str` error is a ValueError, an `io::Error` is an IOError, and returning a `femtopython::Error` keeps its kind
* `error.class()` gives the python-style exception class (`ErrorClass::Type` = TypeError etc) for coarse matching
* `Display` is python-style: `TypeError: ...`. `error.render(src)` adds the line/col and a caret under the source

## Limits
Everything has a limit so a script can't take the host down. All configurable.
* filesystem off by default
* expression nesting (200), type/value nesting (32, json nesting too), indentation depth (64), `map` nesting (32)
* evaluation depth (400, expressions + calls). Deep evaluation grows the stack onto the heap instead of overflowing, so recursion is always an error and never a crash
* steps (statements, loop iterations, iterator items), 10M by default, unlimited in the repl
* string/bytes size from concatenation, file reads and `json.dumps`, 16MiB
* an interrupt flag the host can set from any thread (ctrl-c in the repl)

# Parsing
Input is fully recognized before any of it runs: a regular lexer with a bounded indentation stack, an LR(1) grammar (lalrpop), then a static check (`return` only in functions, `break`/`continue` only in loops, no duplicate params). Python keywords that aren't supported (`import`, `class`, `try`, ...) are reserved and give a clear error.

# Repl
`femtopython` binary, behind the default-on `repl` feature.
* `femtopython` for a `>>>` prompt, `femtopython file.py` to run a file, `-c code` to run code, `-i` to drop into the prompt after
* `open` reads files normally, like python. `--no-fs` turns it off, `--allow-path PATH` (repeatable, file or dir) switches to the allowlist
* `--max-eval-depth`, `--max-steps`
* blocks need a blank line to finish, like python
* echoes the value of expressions that aren't None

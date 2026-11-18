This project is a small implementation of a subset of Python. The goal is embedding in Rust programs, simple tasks like acting as a calculator, and environments where embedding a fully compliant Python interpreter risks security issues. I also want it to be more typed than normal python, closer to rust.

This interpreter should support the following features, and no more.
## Types
### Rust-like primitive types
Primitives should be usable like in python, where they can act as functions. If a into() doesn't exist
#### Integers
* i8,i16,i32,i64,u8,u16,u32,u64
* int is alias to i64
#### Floats
* f32,f64
* float is alias to f64
#### Strings
* str (roughly, rust String. indexing and iterating it should be in terms of chars)
* bytes (roughly, rust &[u8], indexing and iterating should return u8)
#### Lists
* No mixed types
### Dicts
* No mixed key types
* .keys() .values() .items()
### Sets
* No mixed types
### Tuples
* Basically same as rust, but mixed-type tuples can't be cast or iterated.

## Variables / literals
Declare normally: `a = 2` (infer int, float, str, bytes from the const value)
Declare with type: `a: int = 2`
b"" byte strings, "" strings
Redeclaration is okay (like Rust's let) but it's a redeclaration, not an assignment. Can't happen in functions or lambdas.

## Functions
Still first class, as before.
Type definitions not required, but calls are typechecked at call time.
Declarations inside functions shadow

## Operations
All operations should operate basically the same as their equivalent rust primitives
* Arithmetic `+ - * / % **`
* Comparators `> >= == != <= <`
* Logical `and or not`
* Bitwise `& | ^ ~ << >>`
* Assignment `= += -= *= %=`

# Builtins outside of function types
* print(varargs of anything that aren't a function) returns None always, varargs are not normally allowed, this function gets special treatment
* map(f: T -> U, Iterator<Item=T>)
* open(filename: str|Path)
* Path(str) (under the hood this should just be PathBuf basically and only really exist to be used for open and support / sugar)

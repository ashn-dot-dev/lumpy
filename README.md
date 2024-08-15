The Lumpy Programming Language
==============================

Lumpy is a small scripting language with value semantics.

Lumpy features strong dynamic typing, structural equality, assignment by copy,
pass by (copied) value, explicit references, and lightweight polymorphism via
metamaps and operator overloading. Lumpy utilizes a copy-on-write data model
that allows for inexpensive copy operations at runtime, yielding a language
that reads, writes, and feels like efficient pseudocode.

```
println("Hello, world!");

print("\n");

let x = ["foo", {"bar": 123}, "baz"];
let y = x; # x is assigned to y by copy
println("x is " + repr(x));
println("y is " + repr(y));
# x and y are separate values with structural equality
println("x == y is " + repr(x == y));

print("\n");

# updates to x and y do not affect each other, because they are separate values
x[0] = "abc";
y[1]["bar"] = "xyz";
println("x is " + repr(x));
println("y is " + repr(y));
# x and y are no longer structurally equal as their contents' now differ
println("x == y is " + repr(x == y));

print("\n");

let z = ["foo", {"bar": "xyz"}, "baz"];
println("z is " + repr(z));
# y and z are separate values with structural equality
println("y == z is " + repr(y == z));

print("\n");

# metamap overloading the "==" operator
let meta = {
    "==": function(lhs, rhs) {
        return lhs.id == rhs.id;
    },
};
let a = {"id": "bananna", "expiry date": "2024-08-24"};
let b = {"id": "bananna", "expiry date": "2024-08-31"};
setmeta(a.&, meta);
setmeta(b.&, meta);
println("a is " + repr(a));
println("b is " + repr(b));
# a and b are semantically equal according to the overloaded "==" operator even
# though they are not structurally equal
println("a == b is " + repr(a == b));

print("\n");

# function arguments are passed by (copied) value
let f = function(person) {
    println("[within function f] person is " + repr(x));
    person["favorite color"] = "purple";
    println("[within function f] person after modification is " + repr(x));
};

let alice = {"name": "alice", "age": 32};
println("alice before calling f is " + repr(alice));
f(alice); # pass a copy of alice
println("alice after calling f is still " + repr(alice));

print("\n");

let birthday = function(person_reference) {
    person_reference.*["age"] = person_reference.*["age"] + 1;
};

birthday(alice.&); # pass a refrence to alice
println("alice after calling birthday is " + repr(alice));
```

```
/path/to/lumpy$ ./lumpy readme.lumpy
Hello, world!

x is ["foo", {"bar": 123}, "baz"]
y is ["foo", {"bar": 123}, "baz"]
x == y is true

set x[0] to "abc"
set y[1]["bar"] to "xyz"
x is ["abc", {"bar": 123}, "baz"]
y is ["foo", {"bar": "xyz"}, "baz"]
x == y is false

z is ["foo", {"bar": "xyz"}, "baz"]
y == z is true

a is {"id": "bananna", "expiry date": "2024-08-24"}
b is {"id": "bananna", "expiry date": "2024-08-31"}
a == b is true

alice before calling f is {"name": "alice", "age": 32}
[within function f] person is {"name": "alice", "age": 32}
[within function f] person after modification is {"name": "alice", "age": 32, "favorite color": "purple"}
alice after calling f is still {"name": "alice", "age": 32}

alice after calling birthday is {"name": "alice", "age": 33}
```

A brief language overview can be found in `overview.lumpy`, the output of which
can be viewed by running:

```sh
/path/to/lumpy$ ./lumpy overview.lumpy
```

## Development Setup

```sh
/path/to/lumpy$ python3 -m venv .venv-lumpy
/path/to/lumpy$ . ./.venv-lumpy/bin/activate
(.venv-lumpy) /path/to/lumpy$ python3 -m pip install -r requirements-dev.txt
(.venv-lumpy) /path/to/lumpy$ make check   # run tests
(.venv-lumpy) /path/to/lumpy$ make lint    # lint with mypy
(.venv-lumpy) /path/to/lumpy$ make format  # format using black
```

## License
All content in this repository is licensed under the Zero-Clause BSD license.

See LICENSE for more information.

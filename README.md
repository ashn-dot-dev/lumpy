The Lumpy Programming Language
==============================

Lumpy is a small scripting language with value semantics.

Lumpy features strong dynamic typing, structural equality, assignment by copy,
pass by (copied) value, explicit references, and lightweight polymorphism via
metamaps and operator overloading. Lumpy utilizes a copy-on-write data model
that allows for inexpensive copy operations at runtime, yielding a language
that reads, writes, and feels like efficient pseudocode.

Hello World in Lumpy:

```
# examples/hello-world.lumpy

println("Hello, world!");
```

```sh
/path/to/lumpy$ ./lumpy.py examples/hello-world.lumpy
Hello, world!
```

Lumpy uses value semantics, so assignment operations copy the contents or
"value" of an object when performing the assignment. After an assignment
statement such as `a = b`, the objects `a` and `b` will contain separate copies
of the same value. Lumpy also performs equality comparisons based on structural
equality, so if two object have the same contents, then they are considered to
be equal.

```
# examples/value-semantics-and-structural-equality.lumpy

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
```

```sh
/path/to/lumpy$ ./lumpy.py examples/value-semantics-and-structural-equality.lumpy
x is ["foo", {"bar": 123}, "baz"]
y is ["foo", {"bar": 123}, "baz"]
x == y is true

x is ["abc", {"bar": 123}, "baz"]
y is ["foo", {"bar": "xyz"}, "baz"]
x == y is false

z is ["foo", {"bar": "xyz"}, "baz"]
y == z is true
```

Objects in Lumpy have a metamap that may be used to extend the functionality of
builtin types. In this example the `==` and `!=` operators are overloaded to
test for equality between two values based on their `id` fields rather than
testing based on structural equality.

```
# examples/operator-overloading.lumpy

let meta = {
    "==": function(lhs, rhs) {
        return lhs.id == rhs.id;
    },
    "!=": function(lhs, rhs) {
        return lhs.id != rhs.id;
    },
};
let a = {"id": "bananna", "expiry date": "2024-08-24"};
let b = {"id": "bananna", "expiry date": "2024-08-31"};
setmeta(a.&, meta);
setmeta(b.&, meta);
println("a is " + repr(a));
println("b is " + repr(b));
# a and b are semantically equal according to the overloaded "==" and "!="
# operators even though they are not structurally equal
println("a == b is " + repr(a == b));
println("a != b is " + repr(a != b));
```

```sh
/path/to/lumpy$ ./lumpy.py examples/operator-overloading.lumpy
a is {"id": "bananna", "expiry date": "2024-08-24"}
b is {"id": "bananna", "expiry date": "2024-08-31"}
a == b is true
a != b is false
```

Objects are passed by (copied) value to functions, behaving exactly the same as
if they were assigned (i.e. copied due to value semantics) to each parameter.
References are first-class values in Lumpy, and pass-by-reference is achieved
by taking a refrence to a value with the postfix `.&` operator, and then
passing that value to a function. Lumpy has special syntax where
`value.func(args)` implicitly passes a reference to `value` as the first
parameter to metafunction `func`, similar to `this` within non-static member
functions in C++ or `self` within non-static methods in Python. The special
implicit refrence syntax provides a convenient way to support object-oriented
patterns in Lumpy.

```
# examples/pass-by-value-and-pass-by-reference.lumpy

let f = function(person) {
    println("[within function f] person is " + repr(person));
    person["favorite color"] = "purple";
    println("[within function f] person after modification is " + repr(person));
};

let alice = {"name": "alice", "age": 32};
println("alice before calling f is " + repr(alice));
f(alice); # pass a copy of alice
println("alice after calling f is still " + repr(alice));

print("\n");

let birthday = function(person_ref) {
    # dereference `person_ref` and add one to their `"age"` field.
    person_ref.*.age = person_ref.*.age + 1;
};

birthday(alice.&); # pass a refrence to alice
println("alice after calling birthday is " + repr(alice));

print("\n");

let meta = {
    "birthday": function(self) {
        self.*.age = self.*["age"] + 1;
    },
};
setmeta(alice.&, meta);
alice.birthday(); # alice.& is implicitly passed as the first parameter
println("alice after calling birthday (again) is " + repr(alice));
```

```
/path/to/lumpy$ ./lumpy.py examples/pass-by-value-and-pass-by-reference.lumpy
alice before calling f is {"name": "alice", "age": 32}
[within function f] person is {"name": "alice", "age": 32}
[within function f] person after modification is {"name": "alice", "age": 32, "favorite color": "purple"}
alice after calling f is still {"name": "alice", "age": 32}

alice after calling birthday is {"name": "alice", "age": 33}

alice after calling birthday (again) is {"name": "alice", "age": 34}
```

A brief language overview with more information can be found in
`overview.lumpy`, the output of which can be viewed by running:

```sh
/path/to/lumpy$ ./lumpy.py overview.lumpy
```

An example game built in Lumpy using Pygame can be found under the
`examples/minimalist-game-framework` directory, and may be played by running:

```sh
/path/to/lumpy$ ./lumpy.py examples/minimalist-game-framework/game.lumpy
```

## Development Setup

```sh
/path/to/lumpy$ python3 -m venv .venv-lumpy
/path/to/lumpy$ . .venv-lumpy/bin/activate
(.venv-lumpy) /path/to/lumpy$ python3 -m pip install -r requirements.txt
(.venv-lumpy) /path/to/lumpy$ make check   # run tests
(.venv-lumpy) /path/to/lumpy$ make lint    # lint with mypy
(.venv-lumpy) /path/to/lumpy$ make format  # format using black
```

## License
All content in this repository, unless otherwise noted, is licensed under the
Zero-Clause BSD license.

See LICENSE for more information.

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

Lumpy uses value semantics, so assignment operations copy the contents (i.e.
the "value") of an object when executed. After an assignment statement such as
`a = b`, the objects `a` and `b` will contain separate copies of the same
value. Lumpy also performs equality comparisons based on structural equality,
so if two object have the same contents, then they are considered to be equal.

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

Each object in Lumpy has a metamap that may be used to alter and extend its
functionality. In this example, the `==` and `!=` operators are overloaded in
the metamaps of objects `a` and `b`. The overloaded operators test for equality
between these objects based on the objects' `id` fields rather than their
structural identities.

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
by taking a reference to a value with the postfix `.&` operator, and then
passing that value to a function. Lumpy has special syntax where
`value.func(args)` implicitly passes a reference to `value` as the first
argument to metafunction `func`, similar to `this` within non-static member
functions in C++ or `self` within non-static methods in Python. The special
implicit reference syntax provides a convenient way to support object-oriented
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

birthday(alice.&); # pass a reference to alice
println("alice after calling birthday is " + repr(alice));

print("\n");

let meta = {
    "birthday": function(self) {
        self.*.age = self.*["age"] + 1;
    },
};
setmeta(alice.&, meta);
alice.birthday(); # alice.& is implicitly passed as the first argument
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
`examples/minimalist-game-framework` directory. Install Pygame by following the
steps in the "Development Setup" section below, and then run the example game
with:

```sh
(.venv-lumpy) /path/to/lumpy$ ./lumpy.py examples/minimalist-game-framework/game.lumpy
```

## Development Setup

### Unix-Like Platforms

```sh
/path/to/lumpy$ python3 -m venv .venv-lumpy
/path/to/lumpy$ . .venv-lumpy/bin/activate
(.venv-lumpy) /path/to/lumpy$ python3 -m pip install -r requirements.txt
(.venv-lumpy) /path/to/lumpy$ make check   # run tests
(.venv-lumpy) /path/to/lumpy$ make lint    # lint with mypy and flake8
(.venv-lumpy) /path/to/lumpy$ make format  # format using black
(.venv-lumpy) /path/to/lumpy$ make build   # build standalone executable
(.venv-lumpy) /path/to/lumpy$ make install # install standalone Lumpy tooling
```

### Windows

```ps1
PS C:\path\to\lumpy> python -m venv .venv-lumpy
PS C:\path\to\lumpy> .venv-lumpy\Scripts\Activate.ps1
(.venv-lumpy) PS C:\path\to\lumpy> python -m pip install -r .\requirements.txt
(.venv-lumpy) PS C:\path\to\lumpy> .\make.ps1 check   # run tests
(.venv-lumpy) PS C:\path\to\lumpy> .\make.ps1 lint    # lint with mypy and flake8
(.venv-lumpy) PS C:\path\to\lumpy> .\make.ps1 format  # format using black
(.venv-lumpy) PS C:\path\to\lumpy> .\make.ps1 install # install Lumpy tooling
```

## Installing

### Unix-Like Platforms

The `install` target will install standalone Lumpy tooling into the directory
specified by `LUMPY_HOME` (default `$HOME/.lumpy`). Run `make install` with
`LUMPY_HOME` specified as the directory of your choice:

```sh
$ make install                        # Install to the default $HOME/.lumpy
$ make install LUMPY_HOME=/opt/lumpy  # Install to /opt/lumpy
```

Then, add the following snippet to your `.profile`, replacing `$HOME/.lumpy`
with your chosen `LUMPY_HOME` directory if installing to a non-default
`LUMPY_HOME` location:

```sh
export LUMPY_HOME="$HOME/.lumpy"
if [ -e "$LUMPY_HOME/env" ]; then
    . "$LUMPY_HOME/env"
fi
```

Verify that the standalone Lumpy tooling has been successfully installed by
running `lumpy -h`. You may need to source your `.profile` in new shells until
the start of your next login session.

### Windows

Executing the `install` target will install Lumpy tooling into the
`%PROGRAMFILES%` directory (e.g. `C:\Program Files\Lumpy`). Start a PowerShell
session as an administrator and execute:

```ps1
PS C:\path\to\lumpy> .\make.ps1 install
```

Then, add the following snippet to your `$profile`:

```ps1
$env:LUMPY_HOME = "$env:ProgramFiles\Lumpy"
$env:LUMPY_SEARCH_PATH = "$env:LUMPY_HOME\lib"
function lumpy { python "$env:LUMPY_HOME\lumpy.py" @args }
function lumpy-test { python "$env:LUMPY_HOME\bin\lumpy-test.py" @args }
```

Verify that the Lumpy tooling has been successfully installed by starting a new
PowerShell session and running `lumpy -h`.

## License
All content in this repository, unless otherwise noted, is licensed under the
Zero-Clause BSD license.

See LICENSE for more information.

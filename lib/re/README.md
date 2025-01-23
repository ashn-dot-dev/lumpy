# Regular Expression Operations

Wrapper around Python's [`re`](https://docs.python.org/3/library/re.html).

```
let re = import("re");

let result = re::match(`(\w+) (\w+)`, "Isaac Newton, physicist");
dumpln(result.group(2)); # prints "Newton"
```

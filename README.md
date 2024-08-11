The Lumpy Programming Language
==============================

Lumpy is a small scripting language with value semantics.

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
(.venv-lumpy) /path/to/lumpy$ make lint  # lint with mypy
(.venv-lumpy) /path/to/lumpy$ make format  # format using black
```

## License
All content in this repository is licensed under the Zero-Clause BSD license.

See LICENSE for more information.

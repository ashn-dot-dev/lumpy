.POSIX:
.PHONY: build install check lint format clean

LUMPY_HOME = $$HOME/.lumpy

all: lint format check

bin/lumpy: lumpy.py
	python3 -m nuitka lumpy.py \
		--onefile \
		--output-dir="$$(pwd)/bin" \
		--output-filename=lumpy \
		--remove-output

build: bin/lumpy

install: build
	mkdir -p "$(LUMPY_HOME)"
	cp -r bin "$(LUMPY_HOME)"
	cp -r lib "$(LUMPY_HOME)"
	cp lumpy.py env "$(LUMPY_HOME)"

check:
	LUMPY_HOME="$(realpath .)" sh bin/lumpy-test

lint:
	python3 -m mypy --check-untyped-defs lumpy.py

format:
	python3 -m black --line-length=79 lumpy.py

clean:
	rm -f bin/lumpy
	rm -rf __pycache__
	rm -rf .mypy_cache

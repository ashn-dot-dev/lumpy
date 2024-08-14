.POSIX:
.PHONY: check lint format clean

all: lint format check

check:
	sh lumpy-test

lint:
	python3 -m mypy --check-untyped-defs lumpy

format:
	python3 -m black --line-length=79 lumpy

clean:
	rm -rf __pycache__/
	rm -rf .mypy_cache/

$ErrorActionPreference = "Stop"

function rmrf($file) {
    if (Test-Path "$file") {
        Remove-Item -Recursive -Force "$file"
    }
}

function check {
    python bin\lumpy-test.py
}

function lint {
    python -m mypy --check-untyped-defs lumpy.py bin\lumpy-test.py
    python -m flake8 --ignore=E203,E221,E241,E501,W503 lumpy.py bin\lumpy-test.py
}

function format {
    python -m black --line-length=79 lumpy.py bin\lumpy-test.py
}

function clean {
    rmrf .lumpy-history
    rmrf __pycache__
    rmrf .mypy_cache
}

foreach ($item in $Args) {
    switch ($item) {
        "check" {
            check
        }
        "lint" {
            lint
        }
        "format" {
            format
        }
        "clean" {
            clean
        }
        Default {
            throw "unrecognized target '$item'"
        }
    }
}

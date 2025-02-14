$ErrorActionPreference = "Stop"

function rmrf($file) {
    if (Test-Path "$file") {
        Remove-Item -Recursive -Force "$file"
    }
}

function install {
    $installDirectory = "$env:ProgramFiles\Lumpy"
    echo $installDirectory
    New-Item -Type Directory -Force "$installDirectory"
    New-Item -Type Directory -Force "$installDirectory\bin"
    New-Item -Type Directory -Force "$installDirectory\lib"
    Copy-Item -Force lumpy.py "$installDirectory\lumpy.py"
    Copy-Item -Force bin\lumpy-test.py "$installDirectory\bin\lumpy-test.py"
    Copy-Item -Force -Recurse lib\* "$installDirectory\lib\"
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
        "install" {
            install
        }
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

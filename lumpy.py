#!/usr/bin/env python3

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import UserDict, UserList
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters, digits, printable, whitespace
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Self,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
)
import code
import enum
import math
import os
import random
import re
import sys
import traceback


def escape(text: str) -> str:
    MAPPING = {
        "\t": "\\t",
        "\n": "\\n",
        '"': '\\"',
        "\\": "\\\\",
    }
    return "".join([MAPPING.get(c, c) for c in text])


class InvalidFieldAccess(KeyError):
    def __init__(self, value: "Value", field: "Value"):
        self.value = value.copy()
        self.field = field.copy()
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"invalid access into value {self.value} with field {self.field}"
        )


class SharedVectorData(UserList["Value"]):
    def __init__(self, data: Optional[Iterable["Value"]] = None):
        self.uses: int = 0
        super().__init__(data)

    def copy(self) -> "SharedVectorData":
        return SharedVectorData([x.copy() for x in self])


class SharedMapData(UserDict["Value", "Value"]):
    def __init__(self, data: Optional[dict["Value", "Value"]] = None):
        self.uses: int = 0
        super().__init__(data)

    def copy(self) -> "SharedMapData":
        return SharedMapData(
            {k.copy(): v.copy() for k, v in self.data.items()}
        )


class SharedSetData(UserDict["Value", None]):
    def __init__(self, data: Optional[Iterable["Value"]] = None):
        self.uses: int = 0
        if data is not None:
            super().__init__({k: None for k in data})
        else:
            super().__init__(None)

    def insert(self, element: "Value") -> None:
        super().__setitem__(element, None)

    def remove(self, element: "Value") -> None:
        del self[element]

    def copy(self) -> "SharedSetData":
        return SharedSetData([k.copy() for k in self.data.keys()])


ValueType = TypeVar("ValueType", bound="Value")


class Value(ABC):
    meta: Optional["Map"]

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __setitem__(self, key: "Value", value: "Value") -> None:
        raise NotImplementedError()  # optionally overridden by subclasses

    def __getitem__(self, key: "Value") -> "Value":
        raise NotImplementedError()  # optionally overridden by subclasses

    def __delitem__(self, key: "Value") -> None:
        raise NotImplementedError()  # optionally overridden by subclasses

    def __copy__(self):
        return self.copy()

    def metavalue(self, name: "Value") -> Optional["Value"]:
        if self.meta is None:
            return None
        if name not in self.meta:
            return None
        return self.meta[name]

    def metafunction(
        self, name: "Value"
    ) -> Optional[Union["Function", "Builtin"]]:
        if self.meta is None:
            return None
        if name not in self.meta:
            return None
        function = self.meta[name]
        if not isinstance(function, (Builtin, Function)):
            return None
        return function

    @staticmethod
    @abstractmethod
    def type() -> str:
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def cow(self) -> None:
        raise NotImplementedError()


@final
@dataclass
class Null(Value):
    meta: Optional["Map"] = None

    @staticmethod
    def new() -> "Null":
        # Null values are explicitly not given a metamap by default.
        return Null(meta=None)

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __str__(self) -> str:
        return "null"

    @staticmethod
    def type() -> str:
        return "null"

    def copy(self) -> "Null":
        return Null(self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@final
@dataclass
class Boolean(Value):
    data: bool
    meta: Optional["Map"] = None

    @staticmethod
    def new(data: bool) -> "Boolean":
        return Boolean(data, BOOLEAN_META.copy())

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return self.data == other.data

    def __str__(self) -> str:
        return "true" if self.data else "false"

    @staticmethod
    def type() -> str:
        return "boolean"

    def copy(self) -> "Boolean":
        return Boolean(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@final
@dataclass
class Number(Value):
    data: SupportsFloat
    meta: Optional["Map"]

    def __init__(
        self, data: SupportsFloat, meta: Optional["Map"] = None
    ) -> None:
        # PEP 484 specifies that when an argument is annotated as having type
        # `float`, an argument of type `int` is accepted by the type checker.
        # The Lumpy number type is specifically an IEEE-754 double precision
        # floating point number, so a float cast is used to ensure the typed
        # data is actually a Python float.
        self.data = float(data)
        self.meta = meta

    @staticmethod
    def new(data: SupportsFloat) -> "Number":
        return Number(data, NUMBER_META.copy())

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return self.data == other.data

    def __int__(self) -> int:
        return int(float(self.data))

    def __float__(self) -> float:
        return float(self.data)

    def __str__(self) -> str:
        if math.isnan(self.data):
            return "NaN"
        if self.data == +math.inf:
            return "Inf"
        if self.data == -math.inf:
            return "-Inf"
        string = str(self.data)
        dot = string.find(".")
        end = len(string)
        while string[end - 1] == "0":
            end -= 1  # Remove trailing zeros.
        if dot == end - 1:
            end -= 1  # Remove trailing dot.
        return string[0:end]

    @staticmethod
    def type() -> str:
        return "number"

    def copy(self) -> "Number":
        return Number(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@final
@dataclass
class String(Value):
    data: str
    meta: Optional["Map"] = None

    @staticmethod
    def new(data: str) -> "String":
        return String(data, STRING_META.copy())

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return self.data == other.data

    def __str__(self) -> str:
        return f'"{escape(self.data)}"'

    def __contains__(self, item) -> bool:
        return item in self.data

    def __setitem__(self, key: Value, value: Value) -> None:
        if not isinstance(key, Number):
            raise KeyError("attempted string access using non-number key")
        index = float(key.data)
        if not index.is_integer():
            raise KeyError("attempted string access using non-integer number")
        index = int(index)
        if index < 0:
            raise KeyError("attempted string access using a negative index")
        if not (isinstance(value, String) and len(value.data) == 1):
            raise KeyError(
                f"attempted string assignment with non-character value {value}"
            )
        characters = list(self.data)
        characters[index] = value.data
        self.data = "".join(characters)

    def __getitem__(self, key: Value) -> Value:
        if not isinstance(key, Number):
            raise KeyError("attempted string access using non-number key")
        index = float(key.data)
        if not index.is_integer():
            raise KeyError("attempted string access using non-integer number")
        index = int(index)
        if index < 0:
            raise KeyError("attempted string access using a negative index")
        index = int(index)
        return String(self.data[index])

    @staticmethod
    def type() -> str:
        return "string"

    def copy(self) -> "String":
        return String(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@final
@dataclass
class Vector(Value):
    data: SharedVectorData
    meta: Optional["Map"]

    def __init__(
        self,
        data: Optional[Union[SharedVectorData, Iterable[Value]]] = None,
        meta: Optional["Map"] = None,
    ) -> None:
        if data is not None and not isinstance(data, SharedVectorData):
            data = SharedVectorData(data)
        self.data = data if data is not None else SharedVectorData()
        self.data.uses += 1
        self.meta = meta

    @staticmethod
    def new(
        data: Optional[Union[SharedVectorData, Iterable[Value]]] = None
    ) -> "Vector":
        return Vector(data, VECTOR_META.copy())

    def __del__(self):
        assert self.data.uses >= 1
        self.data.uses -= 1

    def __hash__(self) -> int:
        return hash(str(self.data))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        if len(self.data) != len(other.data):
            return False
        for i in range(len(self.data)):
            if self.data[i] != other.data[i]:
                return False
        return True

    def __str__(self) -> str:
        elements = ", ".join([str(x) for x in self.data])
        return f"[{elements}]"

    def __contains__(self, item) -> bool:
        return item in self.data

    def __setitem__(self, key: Value, value: Value) -> None:
        if not isinstance(key, Number):
            raise KeyError("attempted vector access using non-number key")
        index = float(key.data)
        if not index.is_integer():
            raise KeyError("attempted vector access using non-integer number")
        index = int(index)
        if index < 0:
            raise KeyError("attempted vector access using a negative index")
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        self.data.__setitem__(index, value)

    def __getitem__(self, key: Value) -> Value:
        if not isinstance(key, Number):
            raise KeyError("attempted vector access using non-number key")
        index = float(key.data)
        if not index.is_integer():
            raise KeyError("attempted vector access using non-integer number")
        index = int(index)
        if index < 0:
            raise KeyError("attempted vector access using a negative index")
        return self.data.__getitem__(index)

    def __delitem__(self, key: Value) -> None:
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        super().__delitem__(key)

    @staticmethod
    def type() -> str:
        return "vector"

    def copy(self) -> "Vector":
        return Vector(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()
        self.meta = self.meta.copy() if self.meta else None
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1


@final
@dataclass
class Map(Value):
    data: SharedMapData
    meta: Optional["Map"]

    def __init__(
        self,
        data: Optional[Union[SharedMapData, dict[Value, Value]]] = None,
        meta: Optional["Map"] = None,
    ) -> None:
        if data is not None and not isinstance(data, SharedMapData):
            data = SharedMapData(data)
        self.data = data if data is not None else SharedMapData()
        self.data.uses += 1
        self.meta = meta

    @staticmethod
    def new(
        data: Optional[Union[SharedMapData, dict[Value, Value]]] = None
    ) -> "Map":
        return Map(data, MAP_META.copy())

    def __del__(self):
        assert self.data.uses >= 1
        self.data.uses -= 1

    def __hash__(self) -> int:
        return hash(str(self.data))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        if len(self.data) != len(other.data):
            return False
        for k, v in self.data.items():
            if k not in other.data or other.data[k] != v:
                return False
        return True

    def __str__(self) -> str:
        if len(self.data) == 0:
            return "map{}"
        elements = ", ".join(
            [f"{str(k)}: {str(v)}" for k, v in self.data.items()]
        )
        return f"{{{elements}}}"

    def __contains__(self, item) -> bool:
        return item in self.data

    def __setitem__(self, key: Value, value: Value) -> None:
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        try:
            self.data.__setitem__(key, value)
        except KeyError:
            raise InvalidFieldAccess(self, key)

    def __getitem__(self, key: Value) -> Value:
        try:
            return self.data.__getitem__(key)
        except KeyError:
            raise InvalidFieldAccess(self, key)

    def __delitem__(self, key: Value) -> None:
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        try:
            self.data.__delitem__(key)
        except KeyError:
            raise InvalidFieldAccess(self, key)

    @staticmethod
    def type() -> str:
        return "map"

    def copy(self) -> "Map":
        return Map(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1


@final
@dataclass
class Set(Value):
    data: SharedSetData
    meta: Optional["Map"]

    def __init__(
        self,
        data: Optional[Union[SharedSetData, Iterable[Value]]] = None,
        meta: Optional["Map"] = None,
    ) -> None:
        if data is not None and not isinstance(data, SharedSetData):
            data = SharedSetData(data)
        self.data = data if data is not None else SharedSetData()
        self.data.uses += 1
        self.meta = meta

    @staticmethod
    def new(
        data: Optional[Union[SharedSetData, Iterable[Value]]] = None
    ) -> "Set":
        return Set(data, SET_META.copy())

    def __del__(self):
        assert self.data.uses >= 1
        self.data.uses -= 1

    def __hash__(self) -> int:
        return hash(str(self.data))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        if len(self.data) != len(other.data):
            return False
        for k in self.data.keys():
            if k not in other.data:
                return False
        return True

    def __str__(self):
        if len(self.data) == 0:
            return "set{}"
        elements = ", ".join([f"{str(k)}" for k in self.data.keys()])
        return f"{{{elements}}}"

    def __contains__(self, item) -> bool:
        return item in self.data

    def insert(self, element: "Value") -> None:
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        self.data.insert(element)

    def remove(self, element: "Value") -> None:
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1
        self.data.remove(element)

    @staticmethod
    def type() -> str:
        return "set"

    def copy(self) -> "Set":
        return Set(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()
        if self.data.uses > 1:
            self.data.uses -= 1
            self.data = self.data.copy()  # copy-on-write
            self.data.uses += 1


@final
@dataclass
class Reference(Value):
    data: Value
    meta: Optional["Map"] = None

    @staticmethod
    def new(data: Value) -> "Reference":
        return Reference(data, REFERENCE_META.copy())

    def __hash__(self) -> int:
        return hash(id(self.data))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return id(self.data) == id(other.data)

    def __str__(self):
        return f"reference@{hex(id(self.data))}"

    @staticmethod
    def type() -> str:
        return "reference"

    def copy(self) -> "Reference":
        return Reference(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        # We explicitly do *not* copy self.data as the copied data should still
        # point to the original referenced value (the Python `Value` object).
        if self.meta is not None:
            self.meta.cow()


@final
@dataclass
class Function(Value):
    ast: "AstFunction"
    env: "Environment"
    meta: Optional["Map"] = None

    @staticmethod
    def new(ast: "AstFunction", env: "Environment") -> "Function":
        return Function(ast, env, FUNCTION_META.copy())

    def __hash__(self) -> int:
        return hash(id(self.ast)) + hash(id(self.env))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return id(self.ast) == id(other.ast)

    def __str__(self):
        name = self.ast.name.data if self.ast.name is not None else "function"
        ugly = any(c not in ascii_letters + digits + "_" + ":" for c in name)
        name = f'"{escape(name)}"' if ugly else name
        if self.ast.location is not None:
            return f"{name}@[{self.ast.location}]"
        return f"{name}"

    @staticmethod
    def type() -> str:
        return "function"

    def copy(self) -> "Function":
        return Function(
            self.ast, self.env, self.meta.copy() if self.meta else None
        )

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@dataclass
class Builtin(Value):
    meta: Optional["Map"] = None

    def __post_init__(self):
        # Builtins should the name of the builtin as a class property.
        self.name: String

    def __hash__(self) -> int:
        return hash(id(self.function))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return self.function == other.function

    def __str__(self):
        return f"{self.name.data}@builtin"

    @staticmethod
    def type() -> str:
        return "function"

    def copy(self) -> "Builtin":
        return self

    def call(self, arguments: list[Value]) -> Union[Value, "Error"]:
        try:
            result = self.function(arguments)
            return Null.new() if result is None else result
        except Exception as e:
            message = f"{e}"
            if len(message) == 0:
                message = f"encountered exception {type(e).__name__}"
            return Error(None, String(message))

    @staticmethod
    def expect_argument_count(arguments: list[Value], count: int) -> None:
        if len(arguments) != count:
            raise Exception(
                f"invalid argument count (expected {count}, received {len(arguments)})"
            )

    @staticmethod
    def typed_argument(
        arguments: list[Value], index: int, ty: Type[ValueType]
    ) -> ValueType:
        argument = arguments[index]
        if not isinstance(argument, ty):
            raise Exception(
                f"expected {ty.type()}-like value for argument {index}, received {typename(argument)}"
            )
        return argument

    @staticmethod
    def typed_argument_reference(
        arguments: list[Value], index: int, ty: Type[ValueType]
    ) -> Tuple[Reference, ValueType]:
        argument = arguments[index]
        if not (
            isinstance(argument, Reference) and isinstance(argument.data, ty)
        ):
            raise Exception(
                f"expected reference to {ty.type()}-like value for argument {index}, received {typename(argument)}"
            )
        return (argument, argument.data)

    @abstractmethod
    def function(self, arguments: list[Value]) -> Union[Value, "Error"]:
        raise NotImplementedError()

    def cow(self) -> None:
        if self.meta is not None:
            self.meta.cow()


@dataclass
class BuiltinFromSource(Builtin):
    env: Optional["Environment"] = None

    def __post_init__(self) -> None:
        self.evaled = eval_source(self.source(), self.env)
        assert isinstance(self.evaled, Function)

    def __hash__(self) -> int:
        return hash(id(self.evaled))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return self.evaled == other.evaled

    def function(self, arguments: list[Value]) -> Union[Value, "Error"]:
        assert isinstance(self.evaled, Function)
        return call(None, self.evaled, arguments)

    @staticmethod
    @abstractmethod
    def source() -> str:
        raise NotImplementedError()


@dataclass
class External(Value):
    data: Any
    meta: Optional["Map"] = None

    @staticmethod
    def new(data: Any) -> "External":
        return External(data, meta=None)

    def __hash__(self) -> int:
        return hash(id(self.data))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return id(self.data) == id(other.data)

    def __str__(self):
        return f"external({repr(self.data)})"

    @staticmethod
    def type() -> str:
        return f"external"

    def copy(self) -> "External":
        # External values are explicitly not given a metamap by default.
        return External(self.data, self.meta.copy() if self.meta else None)

    def cow(self) -> None:
        # We explicitly do *not* copy self.data as the copied data should still
        # point to the original external Python object.
        if self.meta is not None:
            self.meta.cow()


@dataclass
class SourceLocation:
    filename: Optional[str]
    line: int

    def __str__(self) -> str:
        if self.filename is None:
            return f"line {self.line}"
        return f"{self.filename}, line {self.line}"


class TokenKind(enum.Enum):
    # Meta
    ILLEGAL = "illegal"
    EOF = "eof"
    # Identifiers and Literals
    IDENTIFIER = "identifier"
    NUMBER = "number"
    STRING = "string"
    # Operators
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    REM = "%"
    EQ = "=="
    NE = "!="
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    MKREF = ".&"
    DEREF = ".*"
    DOT = "."
    SCOPE = "::"
    ASSIGN = "="
    # Delimiters
    COMMA = ","
    COLON = ":"
    SEMICOLON = ";"
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    # Keywords
    NULL = "null"
    TRUE = "true"
    FALSE = "false"
    MAP = "map"
    SET = "set"
    NOT = "not"
    AND = "and"
    OR = "or"
    LET = "let"
    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    FOR = "for"
    IN = "in"
    WHILE = "while"
    BREAK = "break"
    CONTINUE = "continue"
    TRY = "try"
    ERROR = "error"
    RETURN = "return"
    FUNCTION = "function"

    def __str__(self) -> str:
        return self.value


@dataclass
class Token:
    KEYWORDS = {
        # fmt: off
        str(TokenKind.NULL):     TokenKind.NULL,
        str(TokenKind.TRUE):     TokenKind.TRUE,
        str(TokenKind.FALSE):    TokenKind.FALSE,
        str(TokenKind.MAP):      TokenKind.MAP,
        str(TokenKind.SET):      TokenKind.SET,
        str(TokenKind.NOT):      TokenKind.NOT,
        str(TokenKind.AND):      TokenKind.AND,
        str(TokenKind.OR):       TokenKind.OR,
        str(TokenKind.LET):      TokenKind.LET,
        str(TokenKind.IF):       TokenKind.IF,
        str(TokenKind.ELIF):     TokenKind.ELIF,
        str(TokenKind.ELSE):     TokenKind.ELSE,
        str(TokenKind.FOR):      TokenKind.FOR,
        str(TokenKind.IN):       TokenKind.IN,
        str(TokenKind.WHILE):    TokenKind.WHILE,
        str(TokenKind.BREAK):    TokenKind.BREAK,
        str(TokenKind.CONTINUE): TokenKind.CONTINUE,
        str(TokenKind.TRY):      TokenKind.TRY,
        str(TokenKind.ERROR):    TokenKind.ERROR,
        str(TokenKind.RETURN):   TokenKind.RETURN,
        str(TokenKind.FUNCTION): TokenKind.FUNCTION,
        # fmt: on
    }

    kind: TokenKind
    literal: str
    location: Optional[SourceLocation] = None
    number: Optional[float] = None
    string: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == TokenKind.EOF:
            return f"end-of-file"
        if self.kind == TokenKind.ILLEGAL:
            prettyable = lambda c: c in printable and c not in whitespace
            prettyrepr = lambda c: c if prettyable(c) else f"{ord(c):#04x}"
            return "".join(map(prettyrepr, self.literal))
        if self.kind == TokenKind.IDENTIFIER:
            return f"{self.literal}"
        if self.kind == TokenKind.NUMBER:
            return f"{self.literal}"
        if self.kind == TokenKind.STRING:
            return f"{self.literal}"
        if self.kind.value in Token.KEYWORDS:
            return self.kind.value
        return f"{self.kind.value}"

    @staticmethod
    def lookup_identifier(identifier: str) -> TokenKind:
        return Token.KEYWORDS.get(identifier, TokenKind.IDENTIFIER)


class Lexer:
    EOF_LITERAL = ""
    RE_IDENTIFIER = re.compile(r"^[a-zA-Z_]\w*", re.ASCII)
    RE_NUMBER_HEX = re.compile(r"^0x[0-9a-fA-F]+", re.ASCII)
    RE_NUMBER_DEC = re.compile(r"^\d+(\.\d+)?", re.ASCII)

    def __init__(
        self, source: str, location: Optional[SourceLocation] = None
    ) -> None:
        self.source: str = source
        # What position does the source "start" being parsed from.
        # None if the source is being lexed in a location-independent manner.
        self.location: Optional[SourceLocation] = location
        self.position: int = 0

    @staticmethod
    def _is_letter(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def _current_character(self) -> str:
        if self.position >= len(self.source):
            return Lexer.EOF_LITERAL
        return self.source[self.position]

    def _peek_character(self) -> str:
        if self.position + 1 >= len(self.source):
            return Lexer.EOF_LITERAL
        return self.source[self.position + 1]

    def _is_eof(self) -> bool:
        return self.position >= len(self.source)

    def _advance_character(self) -> None:
        if self._is_eof():
            return
        if self.location is not None:
            self.location.line += self.source[self.position] == "\n"
        self.position += 1

    def _expect_character(self, character: str) -> None:
        assert len(character) == 1
        current = self._current_character()
        escaped = escape(current)
        if self._is_eof():
            raise ParseError(
                self.location,
                f'expected "{escape(character)}", found end-of-file',
            )
        if current != character:
            raise ParseError(
                self.location,
                f"expected \"{escape(character)}\", found '{escaped}'",
            )
        self._advance_character()

    def _skip_whitespace(self) -> None:
        while not self._is_eof() and self._current_character() in whitespace:
            self._advance_character()

    def _skip_comment(self) -> None:
        if self._current_character() != "#":
            return
        while not self._is_eof() and self._current_character() != "\n":
            self._advance_character()
        self._advance_character()

    def _skip_whitespace_and_comments(self) -> None:
        while not self._is_eof() and (
            self._current_character() in whitespace
            or self._current_character() == "#"
        ):
            self._skip_whitespace()
            self._skip_comment()

    def _new_token(self, kind: TokenKind, literal: str, **kwargs) -> Token:
        return Token(kind, literal, self.location, **kwargs)

    def _lex_keyword_or_identifier(self) -> Token:
        assert Lexer._is_letter(self._current_character())
        match = Lexer.RE_IDENTIFIER.match(self.source[self.position :])
        assert match is not None  # guaranteed by regex
        text = match[0]
        self.position += len(text)
        if text in Token.KEYWORDS:
            return self._new_token(Token.KEYWORDS[text], text)
        return self._new_token(TokenKind.IDENTIFIER, text)

    def _lex_number(self) -> Token:
        assert self._current_character() in digits
        match = Lexer.RE_NUMBER_HEX.match(self.source[self.position :])
        if match is not None:
            text = match[0]
            self.position += len(text)
            return self._new_token(
                TokenKind.NUMBER, text, number=float(int(text, 16))
            )
        match = Lexer.RE_NUMBER_DEC.match(self.source[self.position :])
        assert match is not None  # guaranteed by regex
        text = match[0]
        self.position += len(text)
        return self._new_token(TokenKind.NUMBER, text, number=float(text))

    def _lex_string_character(self) -> str:
        if self._is_eof():
            raise ParseError(
                self.location,
                "expected character, found end-of-file",
            )

        if self._current_character() == "\n":
            raise ParseError(
                self.location,
                "expected character, found newline",
            )

        if not self._current_character().isprintable():
            raise ParseError(
                self.location,
                f"expected printable character, found {hex(ord(self._current_character()))}",
            )

        if self._current_character() == "\\" and self._peek_character() == "t":
            self._advance_character()
            self._advance_character()
            return "\t"

        if self._current_character() == "\\" and self._peek_character() == "n":
            self._advance_character()
            self._advance_character()
            return "\n"

        if self._current_character() == "\\" and self._peek_character() == '"':
            self._advance_character()
            self._advance_character()
            return '"'

        if (
            self._current_character() == "\\"
            and self._peek_character() == "\\"
        ):
            self._advance_character()
            self._advance_character()
            return "\\"

        if self._current_character() == "\\":
            sequence = escape(
                self._current_character() + self._peek_character()
            )
            raise ParseError(
                self.location,
                f'expected escape sequence, found "{sequence}"',
            )

        character = self._current_character()
        self._advance_character()
        return character

    def _lex_string(self) -> Token:
        start = self.position
        self._expect_character('"')
        string = ""
        while not self._is_eof() and self._current_character() != '"':
            string += self._lex_string_character()
        self._expect_character('"')
        literal = self.source[start : self.position]
        return self._new_token(TokenKind.STRING, literal, string=string)

    def _lex_raw_string(self) -> Token:
        start = self.position
        self._expect_character("`")
        string = ""
        if self._current_character() == "`" and self._peek_character() == "`":
            self._advance_character()
            self._advance_character()
            current = self._current_character()
            escaped = escape(current)
            if not self._current_character() == "\n":
                raise ParseError(
                    self.location,
                    f"expected newline, found {escaped}",
                )
            self._advance_character()
            while not self._is_eof() and self._current_character() != "`":
                string += self._current_character()
                self._advance_character()
            self._expect_character("`")
            self._expect_character("`")
            self._expect_character("`")
            literal = self.source[start + 4 : self.position - 3]
        else:
            while not self._is_eof() and self._current_character() != "`":
                string += self._current_character()
                self._advance_character()
            self._expect_character("`")
            literal = self.source[start : self.position]
        return self._new_token(TokenKind.STRING, literal, string=string)

    def next_token(self) -> Token:
        if self.location is not None:
            file = self.location.filename
            line = self.location.line
            self.location = SourceLocation(file, line)
        self._skip_whitespace_and_comments()

        if self._is_eof():
            return self._new_token(TokenKind.EOF, Lexer.EOF_LITERAL)

        # Identifiers, Literals, and Keywords
        if Lexer._is_letter(self._current_character()):
            return self._lex_keyword_or_identifier()
        if self._current_character() in digits:
            return self._lex_number()
        if self._current_character() == '"':
            return self._lex_string()
        if self._current_character() == "`":
            return self._lex_raw_string()

        # Operators
        if self._current_character() == "+":
            self._advance_character()
            return self._new_token(TokenKind.ADD, str(TokenKind.ADD))
        if self._current_character() == "-":
            self._advance_character()
            return self._new_token(TokenKind.SUB, str(TokenKind.SUB))
        if self._current_character() == "*":
            self._advance_character()
            return self._new_token(TokenKind.MUL, str(TokenKind.MUL))
        if self._current_character() == "/":
            self._advance_character()
            return self._new_token(TokenKind.DIV, str(TokenKind.DIV))
        if self._current_character() == "%":
            self._advance_character()
            return self._new_token(TokenKind.REM, str(TokenKind.REM))
        if self._current_character() == "=" and self._peek_character() == "=":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.EQ, str(TokenKind.EQ))
        if self._current_character() == "!" and self._peek_character() == "=":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.NE, str(TokenKind.NE))
        if self._current_character() == "<" and self._peek_character() == "=":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.LE, str(TokenKind.LE))
        if self._current_character() == ">" and self._peek_character() == "=":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.GE, str(TokenKind.GE))
        if self._current_character() == "<":
            self._advance_character()
            return self._new_token(TokenKind.LT, str(TokenKind.LT))
        if self._current_character() == ">":
            self._advance_character()
            return self._new_token(TokenKind.GT, str(TokenKind.GT))
        if self._current_character() == "." and self._peek_character() == "&":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.MKREF, str(TokenKind.MKREF))
        if self._current_character() == "." and self._peek_character() == "*":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.DEREF, str(TokenKind.DEREF))
        if self._current_character() == ".":
            self._advance_character()
            return self._new_token(TokenKind.DOT, str(TokenKind.DOT))
        if self._current_character() == ":" and self._peek_character() == ":":
            self._advance_character()
            self._advance_character()
            return self._new_token(TokenKind.SCOPE, str(TokenKind.SCOPE))
        if self._current_character() == "=":
            self._advance_character()
            return self._new_token(TokenKind.ASSIGN, str(TokenKind.ASSIGN))

        # Delimiters
        if self._current_character() == ",":
            self._advance_character()
            return self._new_token(TokenKind.COMMA, str(TokenKind.COMMA))
        if self._current_character() == ":":
            self._advance_character()
            return self._new_token(TokenKind.COLON, str(TokenKind.COLON))
        if self._current_character() == ";":
            self._advance_character()
            return self._new_token(
                TokenKind.SEMICOLON, str(TokenKind.SEMICOLON)
            )
        if self._current_character() == "(":
            self._advance_character()
            return self._new_token(TokenKind.LPAREN, str(TokenKind.LPAREN))
        if self._current_character() == ")":
            self._advance_character()
            return self._new_token(TokenKind.RPAREN, str(TokenKind.RPAREN))
        if self._current_character() == "{":
            self._advance_character()
            return self._new_token(TokenKind.LBRACE, str(TokenKind.LBRACE))
        if self._current_character() == "}":
            self._advance_character()
            return self._new_token(TokenKind.RBRACE, str(TokenKind.RBRACE))
        if self._current_character() == "[":
            self._advance_character()
            return self._new_token(TokenKind.LBRACKET, str(TokenKind.LBRACKET))
        if self._current_character() == "]":
            self._advance_character()
            return self._new_token(TokenKind.RBRACKET, str(TokenKind.RBRACKET))

        token = self._new_token(TokenKind.ILLEGAL, self._current_character())
        self._advance_character()
        return token


class ParseError(Exception):
    def __init__(self, location: Optional[SourceLocation], why: str) -> None:
        self.location = location
        self.why = why

    def __str__(self) -> str:
        if self.location is None:
            return f"{self.why}"
        return f"[{self.location}] {self.why}"


class Environment:
    @dataclass
    class Lookup:
        value: Value
        store: Map

    def __init__(self, outer: Optional["Environment"] = None) -> None:
        self.outer: Optional["Environment"] = outer
        self.store: Map = Map()

    def let(self, name: String, value: Value) -> None:
        self.store[name] = value

    def get(self, name: String) -> Optional[Value]:
        value = self.store.data.get(name, None)
        if value is None and self.outer is not None:
            return self.outer.get(name)
        return value

    def lookup(self, name: String) -> Optional[Lookup]:
        value = self.store.data.get(name, None)
        if value is None and self.outer is not None:
            return self.outer.lookup(name)
        if value is None:
            return None
        return Environment.Lookup(value, self.store)


@dataclass
class Return:
    value: Value


@dataclass
class Break:
    location: Optional[SourceLocation]
    pass


@dataclass
class Continue:
    location: Optional[SourceLocation]
    pass


@dataclass
class Error:
    @dataclass
    class TraceElement:
        location: Optional[SourceLocation]
        function: Union[Function, Builtin]

    location: Optional[SourceLocation]
    value: Value
    trace: list[TraceElement]

    def __init__(
        self, location: Optional[SourceLocation], value: Union[str, Value]
    ) -> None:
        self.location = location
        self.value = String.new(value) if isinstance(value, str) else value
        self.trace = list()

    def __str__(self):
        if isinstance(self.value, String):
            return f"{self.value.data}"
        return f"{self.value}"


ControlFlow = Union[Return, Break, Continue, Error]


def typename(value: Value) -> str:
    metavalue = value.metavalue(String("type"))
    if metavalue is None:
        return value.type()
    if isinstance(metavalue, String):
        return metavalue.data
    return str(metavalue)


def binary_operator_metafunction(
    lhs: Value, rhs: Value, name: Value
) -> Optional[Union[Function, Builtin]]:
    metafunction = lhs.metafunction(name)
    if metafunction is not None:
        return metafunction
    metafunction = rhs.metafunction(name)
    if metafunction is not None:
        return metafunction
    return None


def update_named_functions(map: "AstMap", prefix: str = ""):
    """
    Update the name values of named functions that are children somewhere in
    this map, either direct map-level value or a decendent of another map.
    """
    for k, v in map.elements:
        if isinstance(k, AstString) and isinstance(v, AstFunction):
            v.name = String(prefix + k.data)
        if isinstance(k, AstString) and isinstance(v, AstMap):
            update_named_functions(v, prefix + k.data + str(TokenKind.SCOPE))


class AstNode(ABC):
    location: Optional[SourceLocation]


class AstExpression(AstNode):
    location: Optional[SourceLocation]

    @abstractmethod
    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        """
        Providing cow=True implies that this expression is being evaluated in
        an lvalue context where copy-on-write is required for a modification of
        the value produced by the eval-chain.
        """
        raise NotImplementedError()


class AstStatement(AstNode):
    location: Optional[SourceLocation]

    @abstractmethod
    def eval(self, env: Environment) -> Optional[ControlFlow]:
        raise NotImplementedError()


@final
@dataclass
class AstProgram(AstNode):
    location: Optional[SourceLocation]
    statements: list[AstStatement]

    def eval(self, env: Environment) -> Optional[Union[Value, Error]]:
        for statement in self.statements:
            result = statement.eval(env)
            if isinstance(result, Return):
                return result.value
            if isinstance(result, Break):
                return Error(
                    self.location, "attempted to break outside of a loop"
                )
            if isinstance(result, Continue):
                return Error(
                    self.location, "attempted to continue outside of a loop"
                )
            if isinstance(result, Error):
                return result
        return None


@final
@dataclass
class AstIdentifier(AstExpression):
    location: Optional[SourceLocation]
    name: String

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        value: Optional[Value] = env.get(self.name)
        if value is None:
            return Error(
                self.location, f"identifier `{self.name.data}` is not defined"
            )
        if cow:
            value.cow()
        return value


@final
@dataclass
class AstNull(AstExpression):
    location: Optional[SourceLocation]

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return Null.new()


@final
@dataclass
class AstBoolean(AstExpression):
    location: Optional[SourceLocation]
    data: bool

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return Boolean.new(self.data)


@final
@dataclass
class AstNumber(AstExpression):
    location: Optional[SourceLocation]
    data: float

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return Number.new(self.data)


@final
@dataclass
class AstString(AstExpression):
    location: Optional[SourceLocation]
    data: str

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return String.new(self.data)


@final
@dataclass
class AstVector(AstExpression):
    location: Optional[SourceLocation]
    elements: list[AstExpression]

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        values: list[Value] = list()
        for x in self.elements:
            result = x.eval(env)
            if isinstance(result, Error):
                return result
            values.append(result.copy())
        return Vector.new(values)


@final
@dataclass
class AstMap(AstExpression):
    location: Optional[SourceLocation]
    elements: list[Tuple[AstExpression, AstExpression]]

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        elements: dict[Value, Value] = dict()
        for k, v in self.elements:
            k_result = k.eval(env)
            if isinstance(k_result, Error):
                return k_result
            v_result = v.eval(env)
            if isinstance(v_result, Error):
                return v_result
            elements[k_result.copy()] = v_result.copy()
        return Map.new(elements)


@final
@dataclass
class AstSet(AstExpression):
    location: Optional[SourceLocation]
    elements: list[AstExpression]

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        elements: list[Value] = list()
        for x in self.elements:
            result = x.eval(env)
            if isinstance(result, Error):
                return result
            elements.append(result.copy())
        return Set.new(elements)


@final
@dataclass
class AstFunction(AstExpression):
    location: Optional[SourceLocation]
    parameters: list[AstIdentifier]
    body: "AstBlock"
    name: Optional[String] = None

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return Function.new(self, env)


@final
@dataclass
class AstGrouped(AstExpression):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        return self.expression.eval(env, cow=cow)


@final
@dataclass
class AstPositive(AstExpression):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        metafunction = result.metafunction(String("unary+"))
        if metafunction is not None:
            return call(self.location, metafunction, [result.copy()])
        if isinstance(result, Number):
            return Number.new(+float(result.data))
        return Error(
            self.location,
            f"attempted unary + operation with type `{typename(result)}`",
        )


@final
@dataclass
class AstNegative(AstExpression):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        metafunction = result.metafunction(String("unary-"))
        if metafunction is not None:
            return call(self.location, metafunction, [result.copy()])
        if not isinstance(result, Number):
            return Error(
                self.location,
                f"attempted unary - operation with type `{typename(result)}`",
            )
        return Number.new(-float(result.data))


@final
@dataclass
class AstNot(AstExpression):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        metafunction = result.metafunction(String("not"))
        if metafunction is not None:
            return call(self.location, metafunction, [result.copy()])
        if not isinstance(result, Boolean):
            return Error(
                self.location,
                f"attempted unary not operation with type `{typename(result)}`",
            )
        return Boolean.new(not result.data)


@final
@dataclass
class AstAnd(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        if isinstance(lhs, Boolean) and not lhs.data:
            return Boolean.new(False)  # short circuit

        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        if isinstance(rhs, Boolean) and not rhs.data:
            return Boolean.new(False)  # short circuit

        metafunction = binary_operator_metafunction(lhs, rhs, String("and"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if isinstance(lhs, Boolean) and isinstance(rhs, Boolean):
            return Boolean.new(lhs.data and rhs.data)
        return Error(
            self.location,
            f"attempted binary and operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstOr(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        if isinstance(lhs, Boolean) and lhs.data:
            return Boolean.new(True)  # short circuit

        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        if isinstance(rhs, Boolean) and rhs.data:
            return Boolean.new(True)  # short circuit

        metafunction = binary_operator_metafunction(lhs, rhs, String("or"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if isinstance(lhs, Boolean) and isinstance(rhs, Boolean):
            return Boolean.new(lhs.data or rhs.data)
        return Error(
            self.location,
            f"attempted binary or operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstEq(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("=="))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) == 0)
        return Boolean.new(lhs == rhs)


@final
@dataclass
class AstNe(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("!="))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) != 0)
        return Boolean.new(lhs != rhs)


@final
@dataclass
class AstLe(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("<="))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) <= 0)
        if isinstance(lhs, Number) and isinstance(rhs, Number):
            return Boolean.new(float(lhs.data) <= float(rhs.data))
        if isinstance(lhs, String) and isinstance(rhs, String):
            return Boolean.new(lhs.data <= rhs.data)
        return Error(
            self.location,
            f"attempted <= operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstGe(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String(">="))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) >= 0)
        if isinstance(lhs, Number) and isinstance(rhs, Number):
            return Boolean.new(float(lhs.data) >= float(rhs.data))
        if isinstance(lhs, String) and isinstance(rhs, String):
            return Boolean.new(lhs.data >= rhs.data)
        return Error(
            self.location,
            f"attempted >= operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstLt(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("<"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) < 0)
        if isinstance(lhs, Number) and isinstance(rhs, Number):
            return Boolean.new(float(lhs.data) < float(rhs.data))
        if isinstance(lhs, String) and isinstance(rhs, String):
            return Boolean.new(lhs.data < rhs.data)
        return Error(
            self.location,
            f"attempted < operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstGt(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String(">"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        metafunction = binary_operator_metafunction(
            lhs, rhs, String("compare")
        )
        if metafunction is not None:
            result = call(
                self.location, metafunction, [lhs.copy(), rhs.copy()]
            )
            if isinstance(result, Error):
                return rhs
            if not isinstance(result, Number):
                return Error(
                    self.location,
                    f"metafunction compare returned non-number value {result}",
                )
            return Boolean.new(float(result.data) > 0)
        if isinstance(lhs, Number) and isinstance(rhs, Number):
            return Boolean.new(float(lhs.data) > float(rhs.data))
        if isinstance(lhs, String) and isinstance(rhs, String):
            return Boolean.new(lhs.data > rhs.data)
        return Error(
            self.location,
            f"attempted > operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstAdd(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("+"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if isinstance(lhs, Number) and isinstance(rhs, Number):
            return Number.new(float(lhs.data) + float(rhs.data))
        if isinstance(lhs, String) and isinstance(rhs, String):
            return String(lhs.data + rhs.data)
        return Error(
            self.location,
            f"attempted + operation with types `{typename(lhs)}` and `{typename(rhs)}`",
        )


@final
@dataclass
class AstSub(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("-"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if not (isinstance(lhs, Number) and isinstance(rhs, Number)):
            return Error(
                self.location,
                f"attempted - operation with types `{typename(lhs)}` and `{typename(rhs)}`",
            )
        return Number.new(float(lhs.data) - float(rhs.data))


@final
@dataclass
class AstMul(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("*"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if not (isinstance(lhs, Number) and isinstance(rhs, Number)):
            return Error(
                self.location,
                f"attempted * operation with types `{typename(lhs)}` and `{typename(rhs)}`",
            )
        return Number.new(float(lhs.data) * float(rhs.data))


@final
@dataclass
class AstDiv(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("/"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if not (isinstance(lhs, Number) and isinstance(rhs, Number)):
            return Error(
                self.location,
                f"attempted / operation with types `{typename(lhs)}` and `{typename(rhs)}`",
            )
        if float(rhs.data) == 0.0:
            return Error(self.location, "division by zero")
        return Number.new(float(lhs.data) / float(rhs.data))


@final
@dataclass
class AstRem(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        lhs = self.lhs.eval(env)
        if isinstance(lhs, Error):
            return lhs
        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        metafunction = binary_operator_metafunction(lhs, rhs, String("%"))
        if metafunction is not None:
            return call(self.location, metafunction, [lhs.copy(), rhs.copy()])
        if not (isinstance(lhs, Number) and isinstance(rhs, Number)):
            return Error(
                self.location,
                f"attempted % operation with types `{typename(lhs)}` and `{typename(rhs)}`",
            )
        if float(rhs.data) == 0.0:
            return Error(self.location, "remainder with divisor zero")
        # The remainder will have the same sign as the dividend.
        # This behavior is identical to C's remainder operator.
        #   +7 % +3 => +1
        #   +7 % -3 => +1
        #   -7 % +3 => -1
        #   -7 % -3 => -1
        return Number.new(math.fmod(float(lhs.data), float(rhs.data)))


@final
@dataclass
class AstFunctionCall(AstExpression):
    location: Optional[SourceLocation]
    function: AstExpression
    arguments: list[AstExpression]

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        self_argument: Optional[Value] = None
        if isinstance(self.function, AstAccessDot):
            # Special case when dot access is used for a function call. An
            # implicit `self` argument is passed by reference to the function.
            store = self.function.store.eval(env)
            if isinstance(store, Error):
                return store
            self_argument = Reference.new(store)
            try:
                function = store[self.function.field.name]
            except (NotImplementedError, IndexError, KeyError):
                function = None
            try:
                if function is None and store.meta is not None:
                    function = store.meta[self.function.field.name]
            except KeyError:
                function = None
            if function is None:
                return Error(
                    self.location,
                    f"invalid method access with name {self.function.field.name}",
                )
        else:
            result = self.function.eval(env)
            if isinstance(result, Error):
                return result
            function = result
        if not isinstance(function, (Function, Builtin)):
            return Error(
                self.location,
                f"attempted to call non-function type `{typename(function)}`",
            )

        arguments: list[Value] = list()
        if self_argument is not None:
            arguments.append(self_argument)
        for argument in self.arguments:
            result = argument.eval(env)
            if isinstance(result, Error):
                return result
            arguments.append(result.copy())
        return call(self.location, function, arguments)


@final
@dataclass
class AstAccessIndex(AstExpression):
    location: Optional[SourceLocation]
    store: AstExpression
    field: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        store = self.store.eval(env, cow=cow)
        if isinstance(store, Error):
            return store
        field = self.field.eval(env)
        if isinstance(field, Error):
            return field
        if isinstance(store, String):
            try:
                return store[field]
            except (NotImplementedError, IndexError, KeyError):
                return Error(
                    self.location, f"invalid string access with index {field}"
                )
        if isinstance(store, Vector):
            try:
                return store[field]
            except (NotImplementedError, IndexError, KeyError):
                return Error(
                    self.location, f"invalid vector access with index {field}"
                )
        if isinstance(store, Map):
            try:
                return store[field]
            except (NotImplementedError, IndexError, KeyError):
                return Error(
                    self.location, f"invalid map access with field {field}"
                )
        return Error(
            self.location,
            f"attempted to access field of type `{typename(store)}` with type `{typename(field)}`",
        )


@final
@dataclass
class AstAccessScope(AstExpression):
    location: Optional[SourceLocation]
    store: AstExpression
    field: AstIdentifier

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        store = self.store.eval(env, cow=cow)
        if isinstance(store, Error):
            return store
        field = self.field.name
        if not isinstance(store, Map):
            return Error(
                self.location,
                f"attempted to access field of type `{typename(store)}`",
            )
        try:
            return store[field]
        except KeyError:
            return Error(
                self.location, f"invalid map access with field {field}"
            )


@final
@dataclass
class AstAccessDot(AstExpression):
    location: Optional[SourceLocation]
    store: AstExpression
    field: AstIdentifier

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        store = self.store.eval(env, cow=cow)
        if isinstance(store, Error):
            return store
        field = self.field.name
        try:
            return store[field]
        except (NotImplementedError, IndexError, KeyError):
            pass
        try:
            if store.meta is not None:
                return store.meta[field]
        except KeyError:
            pass
        return Error(
            self.location, f"invalid {store.type()} access with field {field}"
        )


@final
@dataclass
class AstMkref(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        result = self.lhs.eval(env)
        if isinstance(result, Error):
            return result
        return Reference.new(result)


@final
@dataclass
class AstDeref(AstExpression):
    location: Optional[SourceLocation]
    lhs: AstExpression

    def eval(self, env: Environment, cow: bool = False) -> Union[Value, Error]:
        result = self.lhs.eval(env)
        if isinstance(result, Error):
            return result
        if not isinstance(result, Reference):
            return Error(
                self.location,
                f"attempted dereference of non-reference type `{typename(result)}`",
            )
        return result.data


@final
@dataclass
class AstBlock(AstNode):
    location: Optional[SourceLocation]
    statements: list[AstStatement]

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        env = Environment(env)  # Blocks execute with a new lexical scope.
        for statement in self.statements:
            result = statement.eval(env)
            if isinstance(result, Return):
                return result
            if isinstance(result, Break):
                return result
            if isinstance(result, Continue):
                return result
            if isinstance(result, Error):
                return result
        return None


@final
@dataclass
class AstConditional(AstNode):
    location: Optional[SourceLocation]
    condition: AstExpression
    body: AstBlock

    def exec(self, env: Environment) -> Tuple[Optional[ControlFlow], bool]:
        result = self.condition.eval(env)
        if isinstance(result, Error):
            return (result, False)
        if not isinstance(result, Boolean):
            return (
                Error(
                    self.location,
                    f"conditional with non-boolean type `{typename(result)}`",
                ),
                False,
            )
        if result.data:
            return (self.body.eval(Environment(env)), True)
        return (None, False)

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        (result, executed) = self.exec(env)
        return result


@final
@dataclass
class AstStatementLet(AstStatement):
    location: Optional[SourceLocation]
    identifier: AstIdentifier
    expression: AstExpression

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        env.let(self.identifier.name, result.copy())
        return None


@final
@dataclass
class AstStatementFor(AstStatement):
    location: Optional[SourceLocation]
    identifier_k: AstIdentifier
    identifier_v: Optional[AstIdentifier]
    k_is_reference: bool
    v_is_reference: bool
    collection: AstExpression
    block: AstBlock

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        collection = self.collection.eval(env)
        if isinstance(collection, Error):
            return collection
        collection = collection.copy()

        loop_env = Environment(env)
        if isinstance(collection, Number):
            if self.identifier_v is not None:
                return Error(
                    self.location,
                    f"attempted key-value iteration over type `{typename(collection)}`",
                )
            if self.k_is_reference:
                return Error(
                    self.location,
                    f"cannot use a key-reference over type `{typename(collection)}`",
                )
            if not float(collection.data).is_integer():
                return Error(
                    self.location,
                    f"attempted iteration over non-integer number `{collection}`",
                )
            for i in range(int(float(collection.data))):
                loop_env.let(self.identifier_k.name, Number.new(i))
                result = self.block.eval(loop_env)
                if isinstance(result, Return):
                    return result
                if isinstance(result, Break):
                    return None
                if isinstance(result, Continue):
                    continue
                if isinstance(result, Error):
                    return result
        elif isinstance(collection, Vector):
            if self.identifier_v is not None:
                return Error(
                    self.location,
                    f"attempted key-value iteration over type `{typename(collection)}`",
                )
            for x in collection.data:
                loop_env.let(
                    self.identifier_k.name,
                    Reference.new(x) if self.k_is_reference else x.copy(),
                )
                result = self.block.eval(loop_env)
                if isinstance(result, Return):
                    return result
                if isinstance(result, Break):
                    return None
                if isinstance(result, Continue):
                    continue
                if isinstance(result, Error):
                    return result
        elif isinstance(collection, Map):
            if self.k_is_reference:
                return Error(
                    self.location,
                    f"cannot use a key-reference over type `{typename(collection)}`",
                )
            for k, v in collection.data.items():
                loop_env.let(self.identifier_k.name, k.copy())
                if self.identifier_v is not None:
                    loop_env.let(
                        self.identifier_v.name,
                        Reference.new(v) if self.v_is_reference else v.copy(),
                    )
                result = self.block.eval(loop_env)
                if isinstance(result, Return):
                    return result
                if isinstance(result, Break):
                    return None
                if isinstance(result, Continue):
                    continue
                if isinstance(result, Error):
                    return result
        elif isinstance(collection, Set):
            if self.identifier_v is not None:
                return Error(
                    self.location,
                    f"attempted key-value iteration over type `{typename(collection)}`",
                )
            if self.k_is_reference:
                return Error(
                    self.location,
                    f"cannot use a key-reference over type `{typename(collection)}`",
                )
            for x in collection.data:
                loop_env.let(self.identifier_k.name, x.copy())
                result = self.block.eval(loop_env)
                if isinstance(result, Return):
                    return result
                if isinstance(result, Break):
                    return None
                if isinstance(result, Continue):
                    continue
                if isinstance(result, Error):
                    return result
        else:
            return Error(
                self.location,
                f"attempted iteration over type `{typename(collection)}`",
            )
        return None


@final
@dataclass
class AstStatementWhile(AstStatement):
    location: Optional[SourceLocation]
    expression: AstExpression
    block: AstBlock

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        while True:
            expression = self.expression.eval(env)
            if isinstance(expression, Error):
                return expression
            if not isinstance(expression, Boolean):
                return Error(
                    self.location,
                    f"conditional with non-boolean type `{typename(expression)}`",
                )
            if not expression.data:
                break
            result = self.block.eval(Environment(env))
            if isinstance(result, Return):
                return result
            if isinstance(result, Break):
                return None
            if isinstance(result, Continue):
                continue
            if isinstance(result, Error):
                return result
        return None


@final
@dataclass
class AstStatementBreak(AstStatement):
    location: Optional[SourceLocation]

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        return Break(self.location)


@final
@dataclass
class AstStatementContinue(AstStatement):
    location: Optional[SourceLocation]

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        return Continue(self.location)


@final
@dataclass
class AstStatementIfElifElse(AstStatement):
    location: Optional[SourceLocation]
    conditionals: list[AstConditional]
    else_block: Optional[AstBlock]

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        for conditional in self.conditionals:
            (result, executed) = conditional.exec(env)
            if isinstance(result, Return):
                return result
            if isinstance(result, Break):
                return result
            if isinstance(result, Continue):
                return result
            if isinstance(result, Error):
                return result
            if executed:
                return result
        if self.else_block is not None:
            return self.else_block.eval(env)
        return None


@final
@dataclass
class AstStatementTry(AstStatement):
    location: Optional[SourceLocation]
    try_block: AstBlock
    else_identifier: Optional[AstIdentifier]
    else_block: AstBlock

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        result = self.try_block.eval(env)
        if isinstance(result, Return):
            return result
        if isinstance(result, Break):
            return result
        if isinstance(result, Continue):
            return result
        if isinstance(result, Error):
            env = Environment(env)
            if self.else_identifier is not None:
                env.let(self.else_identifier.name, result.value)
            return self.else_block.eval(env)
        return None


@final
@dataclass
class AstStatementError(AstStatement):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        return Error(self.location, result)


@final
@dataclass
class AstStatementReturn(AstStatement):
    location: Optional[SourceLocation]
    expression: Optional[AstExpression]

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        if self.expression is None:
            return Return(Null.new())
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        return Return(result)


@final
@dataclass
class AstStatementExpression(AstStatement):
    location: Optional[SourceLocation]
    expression: AstExpression

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        result = self.expression.eval(env)
        if isinstance(result, Error):
            return result
        return None


@final
@dataclass
class AstStatementAssignment(AstStatement):
    location: Optional[SourceLocation]
    lhs: AstExpression
    rhs: AstExpression

    def eval(self, env: Environment) -> Optional[ControlFlow]:
        store: Value
        field: Value

        if isinstance(self.lhs, AstIdentifier):
            lookup = env.lookup(self.lhs.name)
            if lookup is None:
                return Error(
                    self.location,
                    f"identifier `{self.lhs.name.data}` is not defined",
                )
            store = lookup.store
            field = self.lhs.name
        elif isinstance(self.lhs, AstAccessIndex):
            lhs_store = self.lhs.store.eval(env, cow=True)
            if isinstance(lhs_store, Error):
                return lhs_store
            lhs_field = self.lhs.field.eval(env)
            if isinstance(lhs_field, Error):
                return lhs_field
            store = lhs_store
            field = lhs_field
        elif isinstance(self.lhs, AstAccessDot):
            lhs_store = self.lhs.store.eval(env, cow=True)
            if isinstance(lhs_store, Error):
                return lhs_store
            lhs_field = self.lhs.field.name
            store = lhs_store
            field = lhs_field
        elif isinstance(self.lhs, AstAccessScope):
            lhs_store = self.lhs.store.eval(env, cow=True)
            if isinstance(lhs_store, Error):
                return lhs_store
            lhs_field = self.lhs.field.name
            store = lhs_store
            field = lhs_field
        else:
            return Error(self.location, "attempted assignment to non-lvalue")

        rhs = self.rhs.eval(env)
        if isinstance(rhs, Error):
            return rhs
        if not isinstance(store, (String, Vector, Map)):
            return Error(
                self.location,
                f"attempted access into type `{typename(store)}` with type `{typename(field)}`",
            )
        try:
            store[field] = rhs.copy()
        except IndexError:
            return Error(
                self.location,
                f"invalid {store.type()} access with index {field}",
            )
        return None


class Precedence(enum.IntEnum):
    # fmt: off
    LOWEST  = enum.auto()
    OR      = enum.auto() # or
    AND     = enum.auto() # and
    COMPARE = enum.auto() # == != <= >= < >
    ADD_SUB = enum.auto() # + -
    MUL_DIV = enum.auto() # * /
    PREFIX  = enum.auto() # +x -x
    POSTFIX = enum.auto() # foo(bar, 123) foo[42] .& .*
    # fmt: on


class Parser:
    ParseNud = Callable[["Parser"], AstExpression]
    ParseLed = Callable[["Parser", AstExpression], AstExpression]

    PRECEDENCES: dict[TokenKind, Precedence] = {
        # fmt: off
        TokenKind.OR:       Precedence.OR,
        TokenKind.AND:      Precedence.AND,
        TokenKind.EQ:       Precedence.COMPARE,
        TokenKind.NE:       Precedence.COMPARE,
        TokenKind.LE:       Precedence.COMPARE,
        TokenKind.GE:       Precedence.COMPARE,
        TokenKind.LT:       Precedence.COMPARE,
        TokenKind.GT:       Precedence.COMPARE,
        TokenKind.ADD:      Precedence.ADD_SUB,
        TokenKind.SUB:      Precedence.ADD_SUB,
        TokenKind.MUL:      Precedence.MUL_DIV,
        TokenKind.DIV:      Precedence.MUL_DIV,
        TokenKind.REM:      Precedence.MUL_DIV,
        TokenKind.LPAREN:   Precedence.POSTFIX,
        TokenKind.LBRACKET: Precedence.POSTFIX,
        TokenKind.DOT:      Precedence.POSTFIX,
        TokenKind.SCOPE:    Precedence.POSTFIX,
        TokenKind.MKREF:    Precedence.POSTFIX,
        TokenKind.DEREF:    Precedence.POSTFIX,
        # fmt: on
    }

    def __init__(self, lexer: Lexer) -> None:
        self.lexer: Lexer = lexer
        self.current_token: Token = Token(TokenKind.ILLEGAL, "DEFAULT CURRENT")
        self.peek_token: Token = Token(TokenKind.ILLEGAL, "DEFAULT PEEK")

        # Read two tokens, so that the current and peek tokens are both set.
        self._advance_token()
        self._advance_token()

        self.parse_nud_functions: dict[TokenKind, Parser.ParseNud] = dict()
        self.parse_led_functions: dict[TokenKind, Parser.ParseLed] = dict()

        self._register_nud(TokenKind.IDENTIFIER, Parser.parse_identifier)
        self._register_nud(TokenKind.NULL, Parser.parse_null)
        self._register_nud(TokenKind.TRUE, Parser.parse_boolean)
        self._register_nud(TokenKind.FALSE, Parser.parse_boolean)
        self._register_nud(TokenKind.NUMBER, Parser.parse_number)
        self._register_nud(TokenKind.STRING, Parser.parse_string)
        self._register_nud(TokenKind.LBRACKET, Parser.parse_vector)
        self._register_nud(TokenKind.MAP, Parser.parse_map_or_set)
        self._register_nud(TokenKind.SET, Parser.parse_map_or_set)
        self._register_nud(TokenKind.LBRACE, Parser.parse_map_or_set)
        self._register_nud(TokenKind.FUNCTION, Parser.parse_function)
        self._register_nud(TokenKind.LPAREN, Parser.parse_grouped)
        self._register_nud(TokenKind.ADD, Parser.parse_positive)
        self._register_nud(TokenKind.SUB, Parser.parse_negative)
        self._register_nud(TokenKind.NOT, Parser.parse_not)

        self._register_led(TokenKind.AND, Parser.parse_and)
        self._register_led(TokenKind.OR, Parser.parse_or)
        self._register_led(TokenKind.EQ, Parser.parse_eq)
        self._register_led(TokenKind.NE, Parser.parse_ne)
        self._register_led(TokenKind.LE, Parser.parse_le)
        self._register_led(TokenKind.GE, Parser.parse_ge)
        self._register_led(TokenKind.LT, Parser.parse_lt)
        self._register_led(TokenKind.GT, Parser.parse_gt)
        self._register_led(TokenKind.ADD, Parser.parse_add)
        self._register_led(TokenKind.SUB, Parser.parse_sub)
        self._register_led(TokenKind.MUL, Parser.parse_mul)
        self._register_led(TokenKind.DIV, Parser.parse_div)
        self._register_led(TokenKind.REM, Parser.parse_rem)
        self._register_led(TokenKind.LPAREN, Parser.parse_function_call)
        self._register_led(TokenKind.LBRACKET, Parser.parse_access_index)
        self._register_led(TokenKind.DOT, Parser.parse_access_dot)
        self._register_led(TokenKind.SCOPE, Parser.parse_access_scope)
        self._register_led(TokenKind.MKREF, Parser.parse_mkref)
        self._register_led(TokenKind.DEREF, Parser.parse_deref)

    def _register_nud(self, kind: TokenKind, parse: "Parser.ParseNud") -> None:
        self.parse_nud_functions[kind] = parse

    def _register_led(self, kind: TokenKind, parse: "Parser.ParseLed") -> None:
        self.parse_led_functions[kind] = parse

    def _advance_token(self) -> Token:
        current_token = self.current_token
        self.current_token = self.peek_token
        self.peek_token = self.lexer.next_token()
        return current_token

    def _check_current(self, kind: TokenKind) -> bool:
        return self.current_token.kind == kind

    def _check_peek(self, kind: TokenKind) -> bool:
        return self.peek_token.kind == kind

    def _expect_current(self, kind: TokenKind) -> Token:
        current = self.current_token
        if current.kind != kind:
            raise ParseError(
                current.location, f"expected {kind}, found {current}"
            )
        self._advance_token()
        return current

    def parse_program(self) -> AstProgram:
        location = self.current_token.location
        statements: list[AstStatement] = list()
        while not self._check_current(TokenKind.EOF):
            statements.append(self.parse_statement())
        return AstProgram(location, statements)

    def parse_expression(
        self, precedence: Precedence = Precedence.LOWEST
    ) -> AstExpression:
        def get_precedence(kind: TokenKind) -> Precedence:
            return Parser.PRECEDENCES.get(kind, Precedence.LOWEST)

        parse_nud = self.parse_nud_functions.get(self.current_token.kind)
        if parse_nud is None:
            raise ParseError(
                self.current_token.location,
                f"expected expression, found {self.current_token}",
            )
        expression = parse_nud(self)
        while precedence < get_precedence(self.current_token.kind):
            parse_led = self.parse_led_functions.get(
                self.current_token.kind, None
            )
            if parse_led is None:
                return expression
            expression = parse_led(self, expression)
        return expression

    def parse_identifier(self) -> AstIdentifier:
        token = self._expect_current(TokenKind.IDENTIFIER)
        return AstIdentifier(token.location, String(token.literal))

    def parse_null(self) -> AstNull:
        location = self._expect_current(TokenKind.NULL).location
        return AstNull(location)

    def parse_boolean(self) -> AstBoolean:
        if self._check_current(TokenKind.TRUE):
            location = self._expect_current(TokenKind.TRUE).location
            return AstBoolean(location, True)
        if self._check_current(TokenKind.FALSE):
            location = self._expect_current(TokenKind.FALSE).location
            return AstBoolean(location, False)
        raise ParseError(
            self.current_token.location,
            f"expected boolean, found {self.current_token}",
        )

    def parse_number(self) -> AstNumber:
        token = self._expect_current(TokenKind.NUMBER)
        assert token.number is not None
        return AstNumber(token.location, token.number)

    def parse_string(self) -> AstString:
        token = self._expect_current(TokenKind.STRING)
        assert token.string is not None
        return AstString(token.location, token.string)

    def parse_vector(self) -> AstVector:
        location = self._expect_current(TokenKind.LBRACKET).location
        elements: list[AstExpression] = list()
        while not self._check_current(TokenKind.RBRACKET):
            if len(elements) != 0:
                self._expect_current(TokenKind.COMMA)
            if self._check_current(TokenKind.RBRACKET):
                break
            elements.append(self.parse_expression())
        self._expect_current(TokenKind.RBRACKET)
        return AstVector(location, elements)

    def parse_map_or_set(self) -> Union[AstMap, AstSet]:
        ParseMapOrSet = enum.Enum("ParseMapOrSet", ["UNKNOWN", "MAP", "SET"])
        map_or_set = ParseMapOrSet.UNKNOWN
        if self._check_current(TokenKind.MAP):
            map_or_set = ParseMapOrSet.MAP
            self._advance_token()
        if self._check_current(TokenKind.SET):
            map_or_set = ParseMapOrSet.SET
            self._advance_token()
        map_elements: list[Tuple[AstExpression, AstExpression]] = list()
        set_elements: list[AstExpression] = list()
        named_functions: set[AstFunction] = set()

        location = self._expect_current(TokenKind.LBRACE).location
        while not self._check_current(TokenKind.RBRACE):
            if len(map_elements) != 0 or len(set_elements):
                self._expect_current(TokenKind.COMMA)
            if self._check_current(TokenKind.RBRACE):
                break

            expression = self.parse_expression()

            if map_or_set == ParseMapOrSet.UNKNOWN:
                if self._check_current(TokenKind.COLON):
                    map_or_set = ParseMapOrSet.MAP
                else:
                    map_or_set = ParseMapOrSet.SET

            assert map_or_set != ParseMapOrSet.UNKNOWN
            match map_or_set:
                case ParseMapOrSet.MAP:
                    self._expect_current(TokenKind.COLON)
                    map_elements.append((expression, self.parse_expression()))
                case ParseMapOrSet.SET:
                    set_elements.append(expression)

        self._expect_current(TokenKind.RBRACE)
        match map_or_set:
            case ParseMapOrSet.UNKNOWN:
                raise ParseError(location, "ambiguous empty map or set")
            case ParseMapOrSet.MAP:
                result = AstMap(location, map_elements)
                update_named_functions(result)
                return result
            case ParseMapOrSet.SET:
                return AstSet(location, set_elements)

    def parse_function(self) -> AstFunction:
        location = self._expect_current(TokenKind.FUNCTION).location
        parameters: list[AstIdentifier] = list()
        self._expect_current(TokenKind.LPAREN)
        while not self._check_current(TokenKind.RPAREN):
            if len(parameters) != 0:
                self._expect_current(TokenKind.COMMA)
            parameters.append(self.parse_identifier())
        self._expect_current(TokenKind.RPAREN)
        body = self.parse_block()
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                if parameters[i].name == parameters[j].name:
                    raise ParseError(
                        parameters[j].location,
                        f"duplicate function paramter `{parameters[i].name.data}`",
                    )
        return AstFunction(location, parameters, body)

    def parse_grouped(self) -> AstGrouped:
        location = self._expect_current(TokenKind.LPAREN).location
        expression = self.parse_expression()
        self._expect_current(TokenKind.RPAREN)
        return AstGrouped(location, expression)

    def parse_positive(self) -> AstPositive:
        location = self._expect_current(TokenKind.ADD).location
        expression = self.parse_expression(Precedence.PREFIX)
        return AstPositive(location, expression)

    def parse_negative(self) -> AstNegative:
        location = self._expect_current(TokenKind.SUB).location
        expression = self.parse_expression(Precedence.PREFIX)
        return AstNegative(location, expression)

    def parse_not(self) -> AstNot:
        location = self._expect_current(TokenKind.NOT).location
        expression = self.parse_expression(Precedence.PREFIX)
        return AstNot(location, expression)

    def parse_and(self, lhs: AstExpression) -> AstAnd:
        location = self._expect_current(TokenKind.AND).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.AND])
        return AstAnd(location, lhs, rhs)

    def parse_or(self, lhs: AstExpression) -> AstOr:
        location = self._expect_current(TokenKind.OR).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.OR])
        return AstOr(location, lhs, rhs)

    def parse_eq(self, lhs: AstExpression) -> AstEq:
        location = self._expect_current(TokenKind.EQ).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.EQ])
        return AstEq(location, lhs, rhs)

    def parse_ne(self, lhs: AstExpression) -> AstNe:
        location = self._expect_current(TokenKind.NE).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.NE])
        return AstNe(location, lhs, rhs)

    def parse_le(self, lhs: AstExpression) -> AstLe:
        location = self._expect_current(TokenKind.LE).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.LE])
        return AstLe(location, lhs, rhs)

    def parse_ge(self, lhs: AstExpression) -> AstGe:
        location = self._expect_current(TokenKind.GE).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.GE])
        return AstGe(location, lhs, rhs)

    def parse_lt(self, lhs: AstExpression) -> AstLt:
        location = self._expect_current(TokenKind.LT).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.LT])
        return AstLt(location, lhs, rhs)

    def parse_gt(self, lhs: AstExpression) -> AstGt:
        location = self._expect_current(TokenKind.GT).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.GT])
        return AstGt(location, lhs, rhs)

    def parse_add(self, lhs: AstExpression) -> AstAdd:
        location = self._expect_current(TokenKind.ADD).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.ADD])
        return AstAdd(location, lhs, rhs)

    def parse_sub(self, lhs: AstExpression) -> AstSub:
        location = self._expect_current(TokenKind.SUB).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.SUB])
        return AstSub(location, lhs, rhs)

    def parse_mul(self, lhs: AstExpression) -> AstMul:
        location = self._expect_current(TokenKind.MUL).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.MUL])
        return AstMul(location, lhs, rhs)

    def parse_div(self, lhs: AstExpression) -> AstDiv:
        location = self._expect_current(TokenKind.DIV).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.DIV])
        return AstDiv(location, lhs, rhs)

    def parse_rem(self, lhs: AstExpression) -> AstRem:
        location = self._expect_current(TokenKind.REM).location
        rhs = self.parse_expression(Parser.PRECEDENCES[TokenKind.REM])
        return AstRem(location, lhs, rhs)

    def parse_function_call(self, lhs: AstExpression) -> AstFunctionCall:
        location = self._expect_current(TokenKind.LPAREN).location
        arguments: list[AstExpression] = list()
        while not self._check_current(TokenKind.RPAREN):
            if len(arguments) != 0:
                self._expect_current(TokenKind.COMMA)
            if self._check_current(TokenKind.RPAREN):
                break
            arguments.append(self.parse_expression())
        self._expect_current(TokenKind.RPAREN)
        return AstFunctionCall(location, lhs, arguments)

    def parse_access_index(self, lhs: AstExpression) -> AstAccessIndex:
        location = self._expect_current(TokenKind.LBRACKET).location
        field = self.parse_expression()
        self._expect_current(TokenKind.RBRACKET)
        return AstAccessIndex(location, lhs, field)

    def parse_access_dot(self, lhs: AstExpression) -> AstAccessDot:
        location = self._expect_current(TokenKind.DOT).location
        field = self.parse_identifier()
        return AstAccessDot(location, lhs, field)

    def parse_access_scope(self, lhs: AstExpression) -> AstAccessScope:
        location = self._expect_current(TokenKind.SCOPE).location
        field = self.parse_identifier()
        return AstAccessScope(location, lhs, field)

    def parse_mkref(self, lhs: AstExpression) -> AstMkref:
        location = self._expect_current(TokenKind.MKREF).location
        return AstMkref(location, lhs)

    def parse_deref(self, lhs: AstExpression) -> AstDeref:
        location = self._expect_current(TokenKind.DEREF).location
        return AstDeref(location, lhs)

    def parse_block(self) -> AstBlock:
        location = self._expect_current(TokenKind.LBRACE).location
        statements: list[AstStatement] = list()
        while not self._check_current(TokenKind.RBRACE):
            statements.append(self.parse_statement())
        self._expect_current(TokenKind.RBRACE)
        return AstBlock(location, statements)

    def parse_statement(self) -> AstStatement:
        if self._check_current(TokenKind.LET):
            return self.parse_statement_let()
        if self._check_current(TokenKind.IF):
            return self.parse_statement_if_elif_else()
        if self._check_current(TokenKind.FOR):
            return self.parse_statement_for()
        if self._check_current(TokenKind.WHILE):
            return self.parse_statement_while()
        if self._check_current(TokenKind.BREAK):
            return self.parse_statement_break()
        if self._check_current(TokenKind.CONTINUE):
            return self.parse_statement_continue()
        if self._check_current(TokenKind.TRY):
            return self.parse_statement_try()
        if self._check_current(TokenKind.ERROR):
            return self.parse_statement_error()
        if self._check_current(TokenKind.RETURN):
            return self.parse_statement_return()
        return self.parse_statement_expression_or_assignment()

    def parse_statement_let(self) -> AstStatementLet:
        location = self._expect_current(TokenKind.LET).location
        identifier = self.parse_identifier()
        self._expect_current(TokenKind.ASSIGN)
        expression = self.parse_expression()
        self._expect_current(TokenKind.SEMICOLON)
        if isinstance(expression, AstFunction):
            expression.name = identifier.name
        if isinstance(expression, AstMap):
            update_named_functions(
                expression, identifier.name.data + str(TokenKind.SCOPE)
            )
        return AstStatementLet(location, identifier, expression)

    def parse_statement_if_elif_else(self) -> AstStatementIfElifElse:
        assert self.current_token.kind == TokenKind.IF
        location = self.current_token.location

        def parse_conditional() -> AstConditional:
            location = self._advance_token().location
            condition = self.parse_expression()
            body = self.parse_block()
            return AstConditional(location, condition, body)

        conditionals: list[AstConditional] = list()
        while self._check_current(
            TokenKind.ELIF if len(conditionals) else TokenKind.IF
        ):
            conditionals.append(parse_conditional())
        if self._check_current(TokenKind.ELSE):
            self._expect_current(TokenKind.ELSE)
            else_block = self.parse_block()
        else:
            else_block = None

        return AstStatementIfElifElse(location, conditionals, else_block)

    def parse_statement_try(self) -> AstStatementTry:
        location = self._expect_current(TokenKind.TRY).location
        try_block = self.parse_block()
        self._expect_current(TokenKind.ELSE)
        if self._check_current(TokenKind.IDENTIFIER):
            else_identifier = self.parse_identifier()
        else:
            else_identifier = None
        else_block = self.parse_block()
        return AstStatementTry(
            location, try_block, else_identifier, else_block
        )

    def parse_statement_error(self) -> AstStatementError:
        location = self._expect_current(TokenKind.ERROR).location
        expression = self.parse_expression()
        self._expect_current(TokenKind.SEMICOLON)
        return AstStatementError(location, expression)

    def parse_statement_for(self) -> AstStatementFor:
        location = self._expect_current(TokenKind.FOR).location
        identifier_k = self.parse_identifier()
        k_is_reference = False
        v_is_reference = False
        if self._check_current(TokenKind.MKREF):
            self._expect_current(TokenKind.MKREF)
            k_is_reference = True
        identifier_v: Optional[AstIdentifier] = None
        if self._check_current(TokenKind.COMMA):
            self._expect_current(TokenKind.COMMA)
            identifier_v = self.parse_identifier()
            if self._check_current(TokenKind.MKREF):
                self._expect_current(TokenKind.MKREF)
                v_is_reference = True
        self._expect_current(TokenKind.IN)
        collection = self.parse_expression()
        block = self.parse_block()
        if identifier_v is not None and identifier_k.name == identifier_v.name:
            raise ParseError(
                identifier_k.location,
                f"duplicate iterator name `{identifier_k.name.data}`",
            )
        return AstStatementFor(
            location,
            identifier_k,
            identifier_v,
            k_is_reference,
            v_is_reference,
            collection,
            block,
        )

    def parse_statement_while(self) -> AstStatementWhile:
        location = self._expect_current(TokenKind.WHILE).location
        expression = self.parse_expression()
        block = self.parse_block()
        return AstStatementWhile(location, expression, block)

    def parse_statement_break(self) -> AstStatementBreak:
        location = self._expect_current(TokenKind.BREAK).location
        self._expect_current(TokenKind.SEMICOLON)
        return AstStatementBreak(location)

    def parse_statement_continue(self) -> AstStatementContinue:
        location = self._expect_current(TokenKind.CONTINUE).location
        self._expect_current(TokenKind.SEMICOLON)
        return AstStatementContinue(location)

    def parse_statement_return(self) -> AstStatementReturn:
        location = self._expect_current(TokenKind.RETURN).location
        expression: Optional[AstExpression] = None
        if not self._check_current(TokenKind.SEMICOLON):
            expression = self.parse_expression()
        self._expect_current(TokenKind.SEMICOLON)
        return AstStatementReturn(location, expression)

    def parse_statement_expression_or_assignment(
        self,
    ) -> Union[AstStatementExpression, AstStatementAssignment]:
        expression = self.parse_expression()
        if not self._check_current(TokenKind.ASSIGN):
            self._expect_current(TokenKind.SEMICOLON)
            return AstStatementExpression(expression.location, expression)
        location = self._expect_current(TokenKind.ASSIGN).location
        rhs = self.parse_expression()
        self._expect_current(TokenKind.SEMICOLON)
        return AstStatementAssignment(location, expression, rhs)


def call(
    location: Optional[SourceLocation],
    function: Union[Function, Builtin],
    arguments: list[Value],
) -> Union[Value, Error]:
    if isinstance(function, Builtin):
        produced = function.call(arguments)
        if isinstance(produced, Error):
            produced.trace.append(Error.TraceElement(location, function))
        return produced
    assert isinstance(function, Function)
    if len(arguments) != len(function.ast.parameters):
        return Error(
            location,
            f"invalid function argument count (expected {len(function.ast.parameters)}, received {len(arguments)})",
        )
    env = Environment(function.env)
    for i in range(len(function.ast.parameters)):
        env.let(function.ast.parameters[i].name, arguments[i])
    result = function.ast.body.eval(env)
    if isinstance(result, Return):
        return result.value
    if isinstance(result, Break):
        return Error(result.location, "attempted to break outside of a loop")
    if isinstance(result, Continue):
        return Error(
            result.location, "attempted to continue outside of a loop"
        )
    if isinstance(result, Error):
        result.trace.append(Error.TraceElement(location, function))
        return result
    return Null.new()


class BuiltinSetmeta(Builtin):
    name = String("setmeta")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0 = Builtin.typed_argument(arguments, 0, Reference)
        if isinstance(arguments[1], Null):
            arg0.data.meta = None
            return Null.new()
        if isinstance(arguments[1], Map):
            arg0.data.meta = arguments[1]
            return Null.new()
        return Error(
            None,
            f"expected null or map-like argument, received {typename(arguments[1])}",
        )


class BuiltinGetmeta(Builtin):
    name = String("getmeta")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        if arguments[0].meta is None:
            return Null.new()
        return arguments[0].meta


class BuiltinUtype(Builtin):
    name = String("utype")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        return String(arguments[0].type())


class BuiltinType(Builtin):
    name = String("type")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        return String(typename(arguments[0]))


class BuiltinRepr(Builtin):
    name = String("repr")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        return String(str(arguments[0]))


class BuiltinDump(Builtin):
    name = String("dump")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        print(arguments[0], end="")
        return Null.new()


class BuiltinDumpln(Builtin):
    name = String("dumpln")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        print(arguments[0], end="\n")
        return Null.new()


class BuiltinPrint(Builtin):
    name = String("print")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        metafunction = arguments[0].metafunction(String("string"))
        if metafunction is not None:
            result = call(None, metafunction, arguments)
            if isinstance(result, Error):
                return result
            if not isinstance(result, String):
                return Error(None, f"metafunction `string` returned {result}")
            print(result.data, end="")
        elif isinstance(arguments[0], String):
            print(arguments[0].data, end="")
        else:
            print(arguments[0], end="")
        return Null.new()


class BuiltinPrintln(Builtin):
    name = String("println")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        metafunction = arguments[0].metafunction(String("string"))
        if metafunction is not None:
            result = call(None, metafunction, arguments)
            if isinstance(result, Error):
                return result
            if not isinstance(result, String):
                return Error(None, f"metafunction `string` returned {result}")
            print(result.data, end="\n")
        elif isinstance(arguments[0], String):
            print(arguments[0].data, end="\n")
        else:
            print(arguments[0], end="\n")
        return Null.new()


class BuiltinBoolean(Builtin):
    name = String("boolean")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        if isinstance(arguments[0], Boolean):
            return arguments[0]
        if isinstance(arguments[0], Number):
            underlying = float(arguments[0].data)
            return Boolean.new(not (math.isnan(underlying) or underlying == 0))
        if isinstance(arguments[0], String) and arguments[0].data == "true":
            return Boolean.new(True)
        if isinstance(arguments[0], String) and arguments[0].data == "false":
            return Boolean.new(False)
        return Error(None, f"cannot convert value {arguments[0]} to boolean")


class BuiltinNumber(Builtin):
    name = String("number")
    # Based on Lexer.RE_NUMBER_HEX and Lexer.RE_NUMBER_HEX with additional
    # end-of-string anchors to ensure the entire input is matched.
    RE_NUMBER_HEX = re.compile(r"^0x[0-9a-fA-F]+$", re.ASCII)
    RE_NUMBER_DEC = re.compile(r"^\d+(\.\d+)?$", re.ASCII)

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        if isinstance(arguments[0], Number):
            return arguments[0]
        if isinstance(arguments[0], Boolean):
            return Number.new(1 if arguments[0].data else 0)
        if isinstance(arguments[0], String):
            try:
                data = arguments[0].data
                if data.startswith("+"):
                    sign = +1
                    data = data[1:]
                elif data.startswith("-"):
                    sign = -1
                    data = data[1:]
                else:
                    sign = +1

                if data == "Inf":
                    return Number.new(sign * math.inf)
                if data == "NaN":
                    return Number.new(sign * math.nan)
                match_hex = BuiltinNumber.RE_NUMBER_HEX.match(data)
                match_dec = BuiltinNumber.RE_NUMBER_DEC.match(data)
                if match_hex is not None or match_dec is not None:
                    return Number.new(sign * float(data))
            except ValueError:
                # Fallthough to end-of-function error case.
                pass
        return Error(None, f"cannot convert value {arguments[0]} to number")


class BuiltinString(Builtin):
    name = String("string")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        metafunction = arguments[0].metafunction(String("string"))
        if metafunction is not None:
            result = call(None, metafunction, arguments)
            if isinstance(result, Error):
                return result
            if not isinstance(result, String):
                return Error(
                    None, f"metafunction `{self.name.data}` returned {result}"
                )
            return result
        if isinstance(arguments[0], String):
            return arguments[0]
        return String(str(arguments[0]))


class BuiltinVector(Builtin):
    name = String("vector")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        if isinstance(arguments[0], String):
            return Vector([String(x) for x in arguments[0].data])
        if isinstance(arguments[0], Vector):
            return arguments[0]
        if isinstance(arguments[0], Map):
            kv = lambda k, v: Vector([k.copy(), v.copy()])
            return Vector([kv(k, v) for k, v in arguments[0].data.items()])
        if isinstance(arguments[0], Set):
            kv = lambda k, v: Vector([k.copy(), v.copy()])
            return Vector([x.copy() for x in arguments[0].data])
        return Error(None, f"cannot convert value {arguments[0]} to vector")


class BuiltinUnion(BuiltinFromSource):
    name = String("union")

    @staticmethod
    def source() -> str:
        return """
        let union = function(a, b) {
            let utype_a = utype(a);
            let utype_b = utype(b);
            if utype_a == "map" and utype_b == "map" {
                let result = map{};
                let insert = getmeta(result)::insert;
                for k, v in a {
                    insert(result.&, k, v);
                }
                for k, v in b {
                    insert(result.&, k, v);
                }
                return result;
            }
            if utype_a == "set" and utype_b == "set" {
                let result = set{};
                let insert = getmeta(result)::insert;
                for element in a {
                    insert(result.&, element);
                }
                for element in b {
                    insert(result.&, element);
                }
                return result;
            }
            error "expected two map-like or two set-like values, received " + type(a) + " and " + type(b);
        };
        return union;
        """


class BuiltinIntersection(BuiltinFromSource):
    name = String("intersection")

    @staticmethod
    def source() -> str:
        return """
        let intersection = function(a, b) {
            let utype_a = utype(a);
            let utype_b = utype(b);
            if utype_a == "map" and utype_b == "map" {
                let result = map{};
                let contains = getmeta(result)::contains;
                let insert = getmeta(result)::insert;
                for k, v in a {
                    if contains(b.&, k) {
                        insert(result.&, k, v);
                    }
                }
                for k, v in b {
                    if contains(a.&, k) {
                        insert(result.&, k, v);
                    }
                }
                return result;
            }
            if utype_a == "set" and utype_b == "set" {
                let result = set{};
                let contains = getmeta(result)::contains;
                let insert = getmeta(result)::insert;
                for element in a {
                    if contains(b.&, element) {
                        insert(result.&, element);
                    }
                }
                for element in b {
                    if contains(a.&, element) {
                        insert(result.&, element);
                    }
                }
                return result;
            }
            error "expected two map-like or two set-like values, received " + type(a) + " and " + type(b);
        };
        return intersection;
        """


class BuiltinDifference(BuiltinFromSource):
    name = String("difference")

    @staticmethod
    def source() -> str:
        return """
        let difference = function(a, b) {
            let utype_a = utype(a);
            let utype_b = utype(b);
            if utype_a == "map" and utype_b == "map" {
                let result = map{};
                let contains = getmeta(result)::contains;
                let insert = getmeta(result)::insert;
                for k, v in a {
                    if not contains(b.&, k) {
                        insert(result.&, k, v);
                    }
                }
                return result;
            }
            if utype_a == "set" and utype_b == "set" {
                let result = set{};
                let contains = getmeta(result)::contains;
                let insert = getmeta(result)::insert;
                for element in a {
                    if not contains(b.&, element) {
                        insert(result.&, element);
                    }
                }
                return result;
            }
            error "expected two map-like or two set-like values, received " + type(a) + " and " + type(b);
        };
        return difference;
        """


class BuiltinAssert(BuiltinFromSource):
    name = String("assert")

    @staticmethod
    def source() -> str:
        return """
        let assert = function(condition) {
            if not condition {
                error "assertion failure";
            }
        };
        return function(condition) {
            try { assert(condition); } else err { error err; }
        };
        """


class BuiltinMin(BuiltinFromSource):
    name = String("min")

    @staticmethod
    def source() -> str:
        return """
        let min = function(a, b) {
            if a <= b {
                return a;
            }
            return b;
        };
        return min;
        """


class BuiltinMax(BuiltinFromSource):
    name = String("max")

    @staticmethod
    def source() -> str:
        return """
        let max = function(a, b) {
            if a >= b {
                return a;
            }
            return b;
        };
        return max;
        """


class BuiltinImport(Builtin):
    name = String("import")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, String)
        env = Environment(BASE_ENVIRONMENT)
        module = env.get(String("module"))
        assert module is not None
        module_directory = module[String("directory")]
        assert isinstance(module_directory, String)
        # Always search the current module directory first
        paths = [module_directory.data]
        LUMPY_SEARCH_PATH = os.environ.get("LUMPY_SEARCH_PATH")
        if LUMPY_SEARCH_PATH is not None:
            paths += LUMPY_SEARCH_PATH.split(":")
        for p in paths:
            path = Path(p + "/" + arg0.data)
            module[String("directory")] = String(
                os.path.dirname(os.path.realpath(path))
            )
            try:
                result = eval_file(path, env)
                break
            except FileNotFoundError:
                pass
        else:
            result = Error(None, f"module {arg0} not found")
        # Always restore module fields
        module[String("directory")] = module_directory
        if isinstance(result, Error):
            return result
        if result is None:
            return Null.new()
        return result


class BuiltinExtend(Builtin):
    name = String("extend")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        source = Builtin.typed_argument(arguments, 0, String).data
        try:
            exec(source, globals())
        except:
            return Error(None, String.new(traceback.format_exc()))
        return Null.new()


class BuiltinFsRead(Builtin):
    name = String("fs::read")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, String)
        with open(arg0.data, "r") as f:
            data = f.read()
        return String(data)


class BuiltinFsWrite(Builtin):
    name = String("fs::write")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0 = Builtin.typed_argument(arguments, 0, String)
        arg1 = Builtin.typed_argument(arguments, 1, String)
        with open(arg0.data, "w") as f:
            data = f.write(arg1.data)
        return Null.new()


class BuiltinFsAppend(Builtin):
    name = String("fs::append")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0 = Builtin.typed_argument(arguments, 0, String)
        arg1 = Builtin.typed_argument(arguments, 1, String)
        with open(arg0.data, "a") as f:
            data = f.write(arg1.data)
        return Null.new()


class BuiltinMathIsNaN(Builtin):
    name = String("math::is_nan")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Boolean.new(math.isnan(arg0.data))


class BuiltinMathIsInf(Builtin):
    name = String("math::is_inf")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Boolean.new(math.isinf(arg0.data))


class BuiltinMathIsInteger(Builtin):
    name = String("math::is_integer")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Boolean.new(float(arg0.data).is_integer())


class BuiltinMathTrunc(Builtin):
    name = String("math::trunc")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.trunc(float(arg0.data)))


class BuiltinMathRound(Builtin):
    name = String("math::round")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(round(float(arg0.data)))


class BuiltinMathFloor(Builtin):
    name = String("math::floor")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.floor(float(arg0.data)))


class BuiltinMathCeil(Builtin):
    name = String("math::ceil")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.ceil(float(arg0.data)))


class BuiltinMathAbs(Builtin):
    name = String("math::abs")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.fabs(arg0.data))


class BuiltinMathPow(Builtin):
    name = String("math::pow")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        return Number.new(math.pow(arg0.data, arg1.data))


class BuiltinMathSqrt(Builtin):
    name = String("math::sqrt")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.sqrt(arg0.data))


class BuiltinMathClamp(BuiltinFromSource):
    name = String("math::clamp")

    @staticmethod
    def source() -> str:
        return """
        let clamp = function(value, min, max) {
            if value < min {
                return min;
            }
            if value > max {
                return max;
            }
            return value;
        };
        return clamp;
        """


class BuiltinMathSin(Builtin):
    name = String("math::sin")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.sin(arg0.data))


class BuiltinMathCos(Builtin):
    name = String("math::cos")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.cos(arg0.data))


class BuiltinMathTan(Builtin):
    name = String("math::tan")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        return Number.new(math.tan(arg0.data))


class BuiltinRandomNumber(Builtin):
    name = String("random::number")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0 = Builtin.typed_argument(arguments, 0, Number)
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        return Number.new(random.uniform(float(arg0.data), float(arg1.data)))


class BuiltinNumberIsNaN(Builtin):
    name = String("is_nan")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Boolean.new(math.isnan(arg0_data.data))


class BuiltinNumberIsInf(Builtin):
    name = String("is_inf")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Boolean.new(math.isinf(arg0_data.data))


class BuiltinNumberIsInteger(Builtin):
    name = String("is_integer")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Boolean.new(float(arg0_data.data).is_integer())


class BuiltinNumberFixed(Builtin):
    name = String("fixed")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        if not float(arg1.data).is_integer():
            return Error(None, f"expected integer, received {arg1}")
        return Number.new(
            round(float(arg0_data.data), ndigits=int(float(arg1.data)))
        )


class BuiltinNumberTrunc(Builtin):
    name = String("trunc")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Number.new(math.trunc(float(arg0_data.data)))


class BuiltinNumberRound(Builtin):
    name = String("round")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Number.new(round(float(arg0_data.data)))


class BuiltinNumberFloor(Builtin):
    name = String("floor")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Number.new(math.floor(float(arg0_data.data)))


class BuiltinNumberCeil(Builtin):
    name = String("ceil")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Number
        )
        return Number.new(math.ceil(float(arg0_data.data)))


class BuiltinStringContains(Builtin):
    name = String("contains")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        return Boolean.new(arg1.data in arg0_data.data)


class BuiltinStringStartsWith(Builtin):
    name = String("starts_with")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        return Boolean.new(arg0_data.data.startswith(arg1.data))


class BuiltinStringEndsWith(Builtin):
    name = String("ends_with")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        return Boolean.new(arg0_data.data.endswith(arg1.data))


class BuiltinStringTrim(Builtin):
    name = String("trim")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        return String.new(arg0_data.data.strip())


class BuiltinStringFind(Builtin):
    name = String("find")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        found = arg0_data.data.find(arg1.data)
        if found == -1:
            return Null.new()
        return Number.new(found)


class BuiltinStringRfind(Builtin):
    name = String("rfind")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        found = arg0_data.data.rfind(arg1.data)
        if found == -1:
            return Null.new()
        return Number.new(found)


class BuiltinStringJoin(Builtin):
    name = String("join")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, Vector)
        s = str()
        for index, value in enumerate(arg1.data):
            if not isinstance(value, String):
                return Error(
                    None,
                    f"expected string-like value for vector element at index {index}, received {typename(value)}",
                )
            if index != 0:
                s += arg0_data.data
            s += value.data
        return String.new(s)


class BuiltinStringSlice(Builtin):
    name = String("slice")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 3)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        if not float(arg1.data).is_integer():
            return Error(None, f"expected integer index, received {arg1}")
        arg2 = Builtin.typed_argument(arguments, 2, Number)
        if not float(arg2.data).is_integer():
            return Error(None, f"expected integer index, received {arg2}")
        bgn = int(float(arg1.data))
        end = int(float(arg2.data))
        if bgn < 0:
            return Error(None, f"slice begin is less than zero")
        if bgn > len(arg0_data.data):
            return Error(
                None, f"slice begin is greater than the string length"
            )
        if end < 0:
            return Error(None, f"slice end is less than zero")
        if end > len(arg0_data.data):
            return Error(None, f"slice end is greater than the string length")
        if end < bgn:
            return Error(None, f"slice end is less than slice begin")
        return String(arg0_data.data[bgn:end])


class BuiltinStringSplit(Builtin):
    name = String("split")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        if len(arg1.data) == 0:
            return Vector.new([String(x) for x in arg0_data.data])
        split = arg0_data.data.split(arg1.data)
        return Vector.new([String(x) for x in split])


class BuiltinStringCut(Builtin):
    name = String("cut")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, String
        )
        arg1 = Builtin.typed_argument(arguments, 1, String)
        found = arg0_data.data.find(arg1.data)
        if found == -1:
            return Null.new()
        prefix = String(arg0_data.data[0:found])
        suffix = String(arg0_data.data[found + len(arg1.data) :])
        return Map.new(
            {
                String("prefix"): prefix,
                String("suffix"): suffix,
            }
        )


class BuiltinVectorCount(Builtin):
    name = String("count")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        return Number.new(len(arg0_data.data))


class BuiltinVectorContains(Builtin):
    name = String("contains")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        return Boolean.new(arguments[1] in arg0_data)


class BuiltinVectorFind(Builtin):
    name = String("find")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        for index, value in enumerate(arg0_data.data):
            if value == arguments[1]:
                return Number.new(index)
        return Null.new()


class BuiltinVectorRfind(Builtin):
    name = String("rfind")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        for index, value in reversed(list(enumerate(arg0_data.data))):
            if value == arguments[1]:
                return Number.new(index)
        return Null.new()


class BuiltinVectorPush(Builtin):
    name = String("push")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        if arg0_data.data.uses > 1:
            arg0_data.cow()  # copy-on-write
        arg0_data.data.append(arguments[1])
        return Null.new()


class BuiltinVectorPop(Builtin):
    name = String("pop")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        if arg0_data.data.uses > 1:
            arg0_data.cow()  # copy-on-write
        arg0_data.data.pop()
        return Null.new()


class BuiltinVectorInsert(Builtin):
    name = String("insert")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 3)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        if not float(arg1.data).is_integer():
            return Error(None, f"expected integer index, received {arg1}")
        if arg0_data.data.uses > 1:
            arg0_data.cow()  # copy-on-write
        arg0_data.data.insert(int(float(arg1.data)), arguments[2])
        return Null.new()


class BuiltinVectorRemove(Builtin):
    name = String("remove")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        if not float(arg1.data).is_integer():
            return Error(None, f"expected integer index, received {arg1}")
        if arg0_data.data.uses > 1:
            arg0_data.cow()  # copy-on-write
        del arg0_data.data[int(float(arg1.data))]
        return Null.new()


class BuiltinVectorSlice(Builtin):
    name = String("slice")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 3)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        arg1 = Builtin.typed_argument(arguments, 1, Number)
        if not float(arg1.data).is_integer():
            return Error(None, f"expected integer index, received {arg1}")
        arg2 = Builtin.typed_argument(arguments, 2, Number)
        if not float(arg2.data).is_integer():
            return Error(None, f"expected integer index, received {arg2}")
        bgn = int(float(arg1.data))
        end = int(float(arg2.data))
        if bgn < 0:
            return Error(None, f"slice begin is less than zero")
        if bgn > len(arg0_data.data):
            return Error(
                None, f"slice begin is greater than the vector length"
            )
        if end < 0:
            return Error(None, f"slice end is less than zero")
        if end > len(arg0_data.data):
            return Error(None, f"slice end is greater than the vector length")
        if end < bgn:
            return Error(None, f"slice end is less than slice begin")
        # Copy underlying data as the update will alter all Python objects
        # holding references to the underlying `SharedVectorData` object.
        underlying = arg0_data.data.copy()
        return Vector.new(SharedVectorData(underlying[bgn:end]))


class BuiltinVectorReversed(Builtin):
    name = String("reversed")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(
            arguments, 0, Vector
        )
        # Copy underlying data as the update will alter all Python objects
        # holding references to the underlying `SharedVectorData` object.
        underlying = arg0_data.data.copy()
        return Vector.new(SharedVectorData(reversed(underlying)))


class BuiltinVectorSorted(BuiltinFromSource):
    name = String("sorted")

    @staticmethod
    def source() -> str:
        return """
        let sort = function(x) {
            if x.count() <= 1 {
                return x;
            }
            let mid = (x.count() / 2).trunc();
            let lo = sort(x.slice(0, mid));
            let hi = sort(x.slice(mid, x.count()));
            let lo_index = 0;
            let hi_index = 0;
            let result = [];
            for _ in x.count() {
                if lo_index == lo.count() {
                    result.push(hi[hi_index]);
                    hi_index = hi_index + 1;
                }
                elif hi_index == hi.count() {
                    result.push(lo[lo_index]);
                    lo_index = lo_index + 1;
                }
                elif lo[lo_index] < hi[hi_index] {
                    result.push(lo[lo_index]);
                    lo_index = lo_index + 1;
                }
                else {
                    result.push(hi[hi_index]);
                    hi_index = hi_index + 1;
                }
            }
            return result;
        };
        return function(self) {
            try { return sort(self.*); } else err { error err; }
        };
        """


class BuiltinMapCount(Builtin):
    name = String("count")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Map)
        return Number.new(len(arg0_data.data))


class BuiltinMapContains(Builtin):
    name = String("contains")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Map)
        return Boolean.new(arguments[1] in arg0_data)


class BuiltinMapInsert(Builtin):
    name = String("insert")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 3)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Map)
        arg0_data[arguments[1]] = arguments[2]
        return Null.new()


class BuiltinMapRemove(Builtin):
    name = String("remove")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Map)
        try:
            del arg0_data[arguments[1]]
        except KeyError:
            pass
        return Null.new()


class BuiltinSetCount(Builtin):
    name = String("count")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 1)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Set)
        return Number.new(len(arg0_data.data))


class BuiltinSetContains(Builtin):
    name = String("contains")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Set)
        return Boolean.new(arguments[1] in arg0_data)


class BuiltinSetInsert(Builtin):
    name = String("insert")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Set)
        arg0_data.insert(arguments[1])
        return Null.new()


class BuiltinSetRemove(Builtin):
    name = String("remove")

    def function(self, arguments: list[Value]) -> Union[Value, Error]:
        Builtin.expect_argument_count(arguments, 2)
        arg0, arg0_data = Builtin.typed_argument_reference(arguments, 0, Set)
        try:
            arg0_data.remove(arguments[1])
        except KeyError:
            pass
        return Null.new()


BASE_ENVIRONMENT = Environment()
BOOLEAN_META = Map()
NUMBER_META = Map()
STRING_META = Map()
VECTOR_META = Map()
MAP_META = Map()
SET_META = Map()
REFERENCE_META = Map()
FUNCTION_META = Map()


def eval_source(
    source: str,
    env: Optional[Environment] = None,
    loc: Optional[SourceLocation] = None,
) -> Optional[Union[Value, Error]]:
    lexer = Lexer(source, loc)
    parser = Parser(lexer)
    program = parser.parse_program()
    return program.eval(env or Environment(BASE_ENVIRONMENT))


def eval_file(
    path: Union[str, os.PathLike],
    env: Optional[Environment] = None,
    argv: Optional[list[str]] = None,
) -> Optional[Union[Value, Error]]:
    with open(path, "r") as f:
        source = f.read()
    return eval_source(source, env, SourceLocation(str(path), 1))


def let_builtin(map: Map, builtin: Builtin) -> None:
    map[builtin.name] = builtin


let_builtin(NUMBER_META, BuiltinNumberIsNaN())
let_builtin(NUMBER_META, BuiltinNumberIsInf())
let_builtin(NUMBER_META, BuiltinNumberIsInteger())
let_builtin(NUMBER_META, BuiltinNumberFixed())
let_builtin(NUMBER_META, BuiltinNumberTrunc())
let_builtin(NUMBER_META, BuiltinNumberRound())
let_builtin(NUMBER_META, BuiltinNumberFloor())
let_builtin(NUMBER_META, BuiltinNumberCeil())

let_builtin(STRING_META, BuiltinStringContains())
let_builtin(STRING_META, BuiltinStringStartsWith())
let_builtin(STRING_META, BuiltinStringEndsWith())
let_builtin(STRING_META, BuiltinStringTrim())
let_builtin(STRING_META, BuiltinStringFind())
let_builtin(STRING_META, BuiltinStringRfind())
let_builtin(STRING_META, BuiltinStringSlice())
let_builtin(STRING_META, BuiltinStringSplit())
let_builtin(STRING_META, BuiltinStringJoin())
let_builtin(STRING_META, BuiltinStringCut())

let_builtin(VECTOR_META, BuiltinVectorCount())
let_builtin(VECTOR_META, BuiltinVectorContains())
let_builtin(VECTOR_META, BuiltinVectorFind())
let_builtin(VECTOR_META, BuiltinVectorRfind())
let_builtin(VECTOR_META, BuiltinVectorPush())
let_builtin(VECTOR_META, BuiltinVectorPop())
let_builtin(VECTOR_META, BuiltinVectorInsert())
let_builtin(VECTOR_META, BuiltinVectorRemove())
let_builtin(VECTOR_META, BuiltinVectorSlice())
let_builtin(VECTOR_META, BuiltinVectorReversed())
let_builtin(VECTOR_META, BuiltinVectorSorted(None, BASE_ENVIRONMENT))

let_builtin(MAP_META, BuiltinMapCount())
let_builtin(MAP_META, BuiltinMapContains())
let_builtin(MAP_META, BuiltinMapInsert())
let_builtin(MAP_META, BuiltinMapRemove())

let_builtin(SET_META, BuiltinSetCount())
let_builtin(SET_META, BuiltinSetContains())
let_builtin(SET_META, BuiltinSetInsert())
let_builtin(SET_META, BuiltinSetRemove())

BASE_ENVIRONMENT.let(String("NaN"), Number.new(float("NaN")))
BASE_ENVIRONMENT.let(String("Inf"), Number.new(float("Inf")))
let_builtin(BASE_ENVIRONMENT.store, BuiltinSetmeta())
let_builtin(BASE_ENVIRONMENT.store, BuiltinGetmeta())
let_builtin(BASE_ENVIRONMENT.store, BuiltinUtype())
let_builtin(BASE_ENVIRONMENT.store, BuiltinType())
let_builtin(BASE_ENVIRONMENT.store, BuiltinRepr())
let_builtin(BASE_ENVIRONMENT.store, BuiltinDump())
let_builtin(BASE_ENVIRONMENT.store, BuiltinDumpln())
let_builtin(BASE_ENVIRONMENT.store, BuiltinPrint())
let_builtin(BASE_ENVIRONMENT.store, BuiltinPrintln())
let_builtin(BASE_ENVIRONMENT.store, BuiltinBoolean())
let_builtin(BASE_ENVIRONMENT.store, BuiltinNumber())
let_builtin(BASE_ENVIRONMENT.store, BuiltinString())
let_builtin(BASE_ENVIRONMENT.store, BuiltinVector())
let_builtin(BASE_ENVIRONMENT.store, BuiltinUnion())
let_builtin(BASE_ENVIRONMENT.store, BuiltinIntersection())
let_builtin(BASE_ENVIRONMENT.store, BuiltinDifference())
let_builtin(BASE_ENVIRONMENT.store, BuiltinAssert())
let_builtin(BASE_ENVIRONMENT.store, BuiltinMin())
let_builtin(BASE_ENVIRONMENT.store, BuiltinMax())
let_builtin(BASE_ENVIRONMENT.store, BuiltinImport())
let_builtin(BASE_ENVIRONMENT.store, BuiltinExtend())
BASE_ENVIRONMENT.let(
    String("fs"),
    Map.new(
        {
            String("read"): BuiltinFsRead(),
            String("write"): BuiltinFsWrite(),
            String("append"): BuiltinFsAppend(),
        }
    ),
)
BASE_ENVIRONMENT.let(
    String("math"),
    Map.new(
        {
            String("pi"): Number.new(math.pi),
            String("is_nan"): BuiltinMathIsNaN(),
            String("is_inf"): BuiltinMathIsInf(),
            String("is_integer"): BuiltinMathIsInteger(),
            String("trunc"): BuiltinMathTrunc(),
            String("round"): BuiltinMathRound(),
            String("floor"): BuiltinMathFloor(),
            String("ceil"): BuiltinMathCeil(),
            String("abs"): BuiltinMathAbs(),
            String("pow"): BuiltinMathPow(),
            String("sqrt"): BuiltinMathSqrt(),
            String("clamp"): BuiltinMathClamp(),
            String("sin"): BuiltinMathSin(),
            String("cos"): BuiltinMathCos(),
            String("tan"): BuiltinMathTan(),
        }
    ),
)
BASE_ENVIRONMENT.let(
    String("module"),
    Map.new(
        {
            String("directory"): Null.new(),
        }
    ),
)
BASE_ENVIRONMENT.let(
    String("random"),
    Map.new(
        {
            String("number"): BuiltinRandomNumber(),
        }
    ),
)


class Repl(code.InteractiveConsole):
    def __init__(self, env: Optional[Environment] = None):
        super().__init__()
        self.env = env if env is not None else Environment(BASE_ENVIRONMENT)

    def runsource(self, source, filename="<input>", symbol="single"):
        lexer = Lexer(source)
        parser = Parser(lexer)
        try:
            program = parser.parse_program()
        except ParseError as e:
            if not source.endswith("\n"):
                # Assume the user has not finished entering their program, and
                # wait for an additional newline before producing an error.
                return True
            print(f"error: {e}")
            return False
        # If the program is valid, but did not end in a semicolon or additional
        # newline, then assume that there may be additonal source to proccess,
        # e.g. the else clause of an if-elif-else statement.
        if not (source.endswith("\n") or source.rstrip().endswith(";")):
            return True
        result = program.eval(self.env)
        if isinstance(result, Value):
            print(result)
        if isinstance(result, Error):
            print(f"error: {result}")
        return False


def main() -> None:
    description = "The Lumpy Programming Language"
    parser = ArgumentParser(description=description)
    parser.add_argument("file", type=str, nargs="?", default=None)
    args, argv = parser.parse_known_args()

    if args.file is not None:
        argv.insert(0, args.file)
        env = Environment(BASE_ENVIRONMENT)
        module = env.get(String("module"))
        assert module is not None
        module[String("directory")] = String(
            os.path.dirname(os.path.realpath(args.file))
        )
        env.let(
            String("argv"),
            Vector([String.new(x) for x in argv]),
        )
        try:
            result = eval_file(args.file, env)
        except Exception as e:
            print(e, file=sys.stderr)
            return
        if isinstance(result, Return):
            print(result.value)
        if isinstance(result, Error):
            if result.location is not None:
                print(f"[{result.location}] error: {result}", file=sys.stderr)
            else:
                print(f"error: {result}", file=sys.stderr)
            for element in result.trace:
                s = f"...within {element.function}"
                if element.location is not None:
                    s += f" called from {element.location}"
                print(s, file=sys.stderr)
    else:
        repl = Repl()
        repl.interact(banner="", exitmsg="")


if __name__ == "__main__":
    main()

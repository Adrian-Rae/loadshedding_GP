from typing import TypeVar, Generic

T = TypeVar("T")
TerminalSetType = 'list[Terminal[T]]'
OperatorSetType = 'list[Operator[T]]'
TotalSetType = 'list[Atom[T]]'


class Atom(Generic[T]):
    def arity(self) -> int:
        return 0

    def instance(self):
        return None


# ===== Terminals =========

class Terminal(Atom, Generic[T]):
    # a terminal cannot have arguments
    def arity(self) -> int:
        return 0

    def instance(self):
        return self

    # TODO: REMOVE THIS
    def __repr__(self):
        return "Terminal"


# ===== Operators =========

class Operator(Atom, Generic[T]):

    # an operator has specified number of arguments
    def __init__(self, arity: int):
        self._arity = arity

    def arity(self) -> int:
        return self._arity

    def instance(self):
        return self

    # TODO: REMOVE THIS
    def __repr__(self):
        return "Operator"

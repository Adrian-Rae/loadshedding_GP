from enum import Enum

from GPParseTree import *
from GPExceptions import InvalidPopulationGenerationMethodException, InvalidPopulationSizeException


class Populator(Generic[T]):

    class Method(Enum):
        GROW = 1
        FULL = 2
        RAMPED = 3

    def __init__(self, size: int, max_depth: int, terminal_set: TerminalSetType, operator_set: OperatorSetType, method: Method = Method.GROW) -> None:

        if method is None:
            raise InvalidPopulationGenerationMethodException
        if size < 1:
            raise InvalidPopulationSizeException
        if max_depth < 2:
            raise InvalidDepthException

        self._size = size
        self._max_depth = max_depth
        self._method = method
        self._terminal_set = terminal_set
        self._operator_set = operator_set

    # TODO: finish FULL and RAMPED
    def generate(self) -> list[ParseTree[T]]:
        if self._method == Populator.Method.GROW:
            return [ParseTree.random(self._max_depth,self._terminal_set,self._operator_set) for i in range(self._size)]
        elif self._method == Populator.Method.FULL:
            return []
        elif self._method == Populator.Method.RAMPED:
            return []
        raise InvalidPopulationGenerationMethodException

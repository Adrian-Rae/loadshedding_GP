from enum import Enum

from GPExceptions import InvalidPopulationGenerationMethodException, InvalidPopulationSizeException
from GPParseTree import *


class Populator(Generic[T]):

    class Method(Enum):
        GROW = 1
        FULL = 2
        RAMPED = 3

    def __init__(self, size: int, max_depth: int, terminal_set: TerminalSetType, operator_set: OperatorSetType,
                 method: Method = Method.GROW) -> None:

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
    def generate(self) -> List[ParseTree[T]]:

        # GROW METHOD
        if self._method == Populator.Method.GROW:

            # Simply return n randomly generated trees with the specified maximum depth
            return [ParseTree.random(self._max_depth, self._terminal_set, self._operator_set) for _ in
                    range(self._size)]

        # FULL METHOD
        elif self._method == Populator.Method.FULL:

            # Generate a population of n members with identical (maximum) depth
            population: List[ParseTree[T]] = []

            while True:

                # Create a new member
                new_member: ParseTree[T] = ParseTree.random(self._max_depth, self._terminal_set, self._operator_set)

                # If the depth is desired
                if new_member.get_depth() == self._max_depth:

                    # keep the member
                    population.append(new_member)

                # If the population size is reached, terminate
                if len(population) == self._size:
                    break

            # Return the population
            return population

        # RAMPED HALF-AND-HALF
        elif self._method == Populator.Method.RAMPED:
            return []

        # Invalid generation method bound
        raise InvalidPopulationGenerationMethodException

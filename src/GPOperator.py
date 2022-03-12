from typing import List

from src.GPParseTree import ParseTree
from src.GPSelector import Selector


class GeneticOperator:

    def __init__(self, selector: Selector):
        self._selector = selector

    def reproduce(self, population: List[ParseTree]) -> ParseTree:
        return self._selector.select(population)

    def crossover(self, population: List[ParseTree]) -> ParseTree:
        pass

    def mutation(self, population: List[ParseTree]) -> ParseTree:
        pass

    def permutation(self, population: List[ParseTree]) -> ParseTree:
        pass

    def editing(self, population: List[ParseTree]) -> ParseTree:
        pass

    def encapsulation(self, population: List[ParseTree]) -> ParseTree:
        pass

    def decimation(self, population: List[ParseTree]) -> ParseTree:
        pass

    def inversion(self, population: List[ParseTree]) -> ParseTree:
        pass

    def hoist(self, population: List[ParseTree]) -> ParseTree:
        pass

    def create(self, population: List[ParseTree]) -> ParseTree:
        pass

    def compress(self, population: List[ParseTree]) -> ParseTree:
        pass

    def expand(self, population: List[ParseTree]) -> ParseTree:
        pass



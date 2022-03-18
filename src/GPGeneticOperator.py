from enum import Enum
from typing import List, Tuple, Callable

from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator
from src.GPSelector import Selector


class GeneticOperatorType(Enum):
    REPRODUCTION = 0
    CROSSOVER = 1
    MUTATION = 2


class GeneticOperatorSet:

    default_set = [
        GeneticOperatorType.REPRODUCTION,
        GeneticOperatorType.CROSSOVER,
        GeneticOperatorType.MUTATION
    ]

    def __init__(self, selector: Selector, generator: PopulationGenerator):
        self._selector = selector
        self._generator = generator
        self._operator_index = [
            self.reproduce,
            self.crossover,
            self.mutation
        ]

    def operate(self, population: List[ParseTree], operator: GeneticOperatorType = GeneticOperatorType.REPRODUCTION):
        return self._operator_index[operator.value](population)

    def reproduce(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        return self._selector.select(population).copy(),

    def crossover(self, population: List[ParseTree], max_depth=None) -> Tuple[ParseTree, ParseTree]:

        # choose a threshold to generate trees under
        timeout = 10
        counter = 0

        while counter < timeout:

            # Select two parents from the population to form children
            parent1 = self._selector.select(population)
            parent2 = self._selector.select(population)

            child1 = parent1.copy()
            child2 = parent2.copy()

            # if no maximum depth specified, treat it as the deeper of the two children
            max_depth = max(child1.get_depth(), child2.get_depth()) if max_depth is None else max_depth

            # get two subtrees from either tree, that are not their roots
            subtree1 = child1.random_node()
            subtree2 = child2.random_node()

            # swap subtrees
            child1.replace_node(subtree1, subtree2)
            child2.replace_node(subtree2, subtree1)

            # if depth requirement is satisfied, return
            if child1.get_depth() <= max_depth and child2.get_depth() <= max_depth:
                return child1, child2

            counter += 1

        # else just return two of the same tree to represent a trivial crossover
        d1 = self.reproduce(population)[0]
        d2 = d1.copy()
        return d1, d2

    def mutation(self, population: List[ParseTree], max_depth=None) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # get a random subtree
        removed_subtree = child.random_node(exclude_root=True)

        # the max depth is either specified, or treated as the current child max depth
        max_depth = child.get_depth() if max_depth is None else max_depth

        # depth of node to be removed
        node_depth = child.depth_of(removed_subtree)

        # max depth of subtree to generate
        subtree_max_depth = max_depth - node_depth + 1

        # generate a new subtree - allow trivial trees if necessary
        new_subtree = self._generator.generate(1, subtree_max_depth, force_trivial=True)[0].get_root()
        child.replace_node(removed_subtree, new_subtree)
        return child,

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

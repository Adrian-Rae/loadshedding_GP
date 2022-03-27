from enum import Enum
from typing import List, Tuple

from GPAtom import Constant
from GPParseTree import ParseTree
from GPPopulationGenerator import PopulationGenerator
from GPSelector import Selector


class GeneticOperatorType(Enum):
    REPRODUCTION = 0
    CROSSOVER = 1
    MUTATION = 2
    PERMUTATION = 3
    EDITING = 4
    ENCAPSULATION = 5
    DECIMATION = 6
    INVERSION = 7
    HOIST = 8
    CREATE = 9
    COMPRESS = 10
    EXPAND = 11


class GeneticOperatorSet:
    default_set = [
        (GeneticOperatorType.REPRODUCTION, 1),
        (GeneticOperatorType.CROSSOVER, 1),
        (GeneticOperatorType.MUTATION, 1)
    ]

    def __init__(self, selector: Selector, generator: PopulationGenerator):
        self._selector = selector
        self._generator = generator
        self._operator_index = [
            self.reproduce,
            self.crossover,
            self.mutation,
            self.permutation,
            self.editing,
            self.encapsulation,
            self.decimation,
            self.inversion,
            self.hoist,
            self.create,
            self.compress,
            self.expand
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
        removed_subtree = child.random_node()

        # the max depth is either specified, or treated as the current child max depth
        max_depth = child.get_depth() if max_depth is None else max_depth

        # depth of node to be removed
        node_depth = child.depth_of(removed_subtree)

        # max depth of subtree to generate
        subtree_max_depth = max_depth - node_depth + 1

        # generate a new subtree - allow trivial trees if necessary
        trivial = True if subtree_max_depth < 2 else False
        new_subtree = self._generator.generate(1, subtree_max_depth, force_trivial=trivial)[0].get_root()
        child.replace_node(removed_subtree, new_subtree)
        return child,

    def permutation(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # permute a functional node's children
        child.permutation()

        # return the child
        return child,

    def editing(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # get a non-parameterized node
        node = child.random_node(non_terminal=True, non_parameterized=True)

        # check if there are reductions possible, else just reproduce
        if node is None:
            return child,

        # replace the subtree rooted at node with a constant
        child.replace_node(node, ParseTree.Node(Constant(node.eval())))

        return child,

    def encapsulation(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # TODO: implement

        return child,

    def decimation(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # TODO: implement

        return child,

    def inversion(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        child.swap_random_node_pair()

        return child,

    def hoist(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # get a functional node
        functional_node = child.random_node(non_terminal=True)

        # hoist the new node
        hoisted = ParseTree.hoist(functional_node) if functional_node is not None else child

        return hoisted,

    def create(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # TODO: implement

        return child,

    def compress(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # TODO: implement

        return child,

    def expand(self, population: List[ParseTree]) -> Tuple[ParseTree]:
        # get a parent, copy it to child
        parent = self._selector.select(population)
        child = parent.copy()

        # TODO: implement

        return child,

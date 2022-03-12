import math
import random
from typing import List

from src.GPAtom import Terminal, Operator, Variable, ConstantRange, Constant
from src.GPFitnessFunction import FitnessFunction, FitnessMeasure
from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator
from src.GPSelector import Selector, SelectionMethod


def main():
    # Establish the seed - leave commented for pure random
    seed = 1
    random.seed(seed)

    # Global Properties
    population_size = 1
    max_depth = 2
    generation_method = PopulationGenerator.Method.FULL
    selection_method = SelectionMethod.TOURNAMENT

    # Establish terminals and Operators
    x: Variable = Variable("x")
    alpha: ConstantRange = ConstantRange(-1, 1)
    one: Constant = Constant(1)

    mult: Operator = Operator("*", lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", lambda a, b: a + b, rep="({} + {})")
    minus: Operator = Operator("-", lambda a, b: a - b, rep="({} - {})")
    floor: Operator = Operator("floor", lambda a: math.floor(a), rep="floor({})")

    # Create a terminal and operator set
    t_set: List[Terminal] = [x, one]
    o_set: List[Operator] = [mult, minus]

    # Create a population
    generator: PopulationGenerator = PopulationGenerator(t_set, o_set)
    population: List[ParseTree] = generator.generate(population_size, max_depth, generation_method)

    fitness = FitnessFunction()
    cases = [
        ({"x": 1}, 0),
        ({"x": 2}, 3),
        ({"x": 3}, 8),
        ({"x": 4}, 15),
        ({"x": 5}, 24)
    ]
    for args, target in cases:
        fitness.bind_case(args, target)

    # # Do things with the population
    # for i, k in enumerate(population):
    #     r = k.random_node()
    #     print("Tree {} [Fitness: {}][Random Node: {}][Lineage of random: {}]: {}".format(
    #         i,
    #         fitness.fitness(k, measure=FitnessMeasure.NORMALIZED, population=population),
    #         r.eval(symbolic=True),
    #         k.get_node_lineage(r),
    #         k.eval(symbolic=True)
    #     ))

    ind = population[0]
    print(ind)
    n1, n2 = ind.random_node_pair()
    print(n1, n2)
    ind.swap_nodes(n1, n2)
    print(ind)


if __name__ == '__main__':
    main()

import math
import random
from typing import List

from GPAtom import Terminal, Operator, Variable, Constant, ConstantRange, Atom
from GPPopulationGenerator import PopulationGenerator
from src.GPParseTree import ParseTree


def main():
    # Establish the seed - leave commented for pure random
    seed = 100
    random.seed(seed)

    # Global Properties
    population_size = 10
    max_depth = 5
    generation_method = PopulationGenerator.Method.RAMPED

    # Establish terminals and Operators
    x: Variable = Variable("x", 5)
    y: Variable = Variable("y", 4)
    alpha: ConstantRange = ConstantRange(1, 5)

    mult: Operator = Operator("*", 2, lambda a, b: a * b)
    add: Operator = Operator("+", 2, lambda a, b: a + b)
    floor: Operator = Operator("floor", 1, lambda a: math.floor(a))

    # Create a terminal and operator set
    t_set: List[Terminal] = [x, y, alpha]
    o_set: List[Operator] = [mult, add, floor]

    # Create a population
    generator: PopulationGenerator = PopulationGenerator(t_set, o_set)
    population: List[ParseTree] = generator.generate(population_size, max_depth, generation_method)

    # Do things with the population
    for k in population:
        print("Tree {}:\n{}".format(k.eval(), k))


if __name__ == '__main__':
    main()

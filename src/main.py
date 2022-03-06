import math
import random
import sys
from typing import List

from GPAtom import Terminal, Operator, Variable, Constant, ConstantRange, Atom
from GPPopulationGenerator import PopulationGenerator
from src.GPParseTree import ParseTree


def main():
    # Establish the seed - leave commented for pure random
    seed = 1
    random.seed(seed)

    # Global Properties
    population_size = 3
    max_depth = 10
    generation_method = PopulationGenerator.Method.RAMPED

    # Establish terminals and Operators
    x: Variable = Variable("x")
    Variable("y", 4)
    alpha: ConstantRange = ConstantRange(-100, 100)

    mult: Operator = Operator("*", 2, lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", 2, lambda a, b: a + b, rep="({} + {})")
    floor: Operator = Operator("floor", 1, lambda a: math.floor(a), rep="floor({})")

    # Create a terminal and operator set
    t_set: List[Terminal] = [x, alpha]
    o_set: List[Operator] = [mult, add, floor]

    # Create a population
    generator: PopulationGenerator = PopulationGenerator(t_set, o_set)
    population: List[ParseTree] = generator.generate(population_size, max_depth, generation_method)

    # Do things with the population
    for i, k in enumerate(population):
        print("Tree {}: {} = {}".format(i, k.eval(x=5), k.eval(symbolic=True)))


if __name__ == '__main__':
    main()

import math
import random
from inspect import signature
from typing import List

from src.GPAtom import Terminal, Operator, Variable, ConstantRange
from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator


def main():
    # Establish the seed - leave commented for pure random
    seed = 1
    # random.seed(seed)

    # Global Properties
    population_size = 3
    max_depth = 10
    generation_method = PopulationGenerator.Method.RAMPED

    # Establish terminals and Operators
    x: Variable = Variable("x")
    alpha: ConstantRange = ConstantRange(-100, 100)

    mult: Operator = Operator("*", lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", lambda a, b: a + b, rep="({} + {})")
    floor: Operator = Operator("floor", lambda a: math.floor(a), rep="floor({})")

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

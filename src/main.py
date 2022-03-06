import random
from typing import List

from GPAtom import Terminal, Operator
from GPPopulationGenerator import PopulationGenerator
from src.GPParseTree import ParseTree


def main():
    # Global Properties
    seed = 100
    population_size = 10
    max_depth = 5
    generation_method = PopulationGenerator.Method.RAMPED

    # Establish terminals and Operators

    # Create a terminal and operator set
    t_set: List[Terminal] = [Terminal(), Terminal()]
    o_set: List[Operator] = [Operator(1), Operator(3)]

    # Establish the seed
    random.seed(seed)

    # Create a population
    generator: PopulationGenerator = PopulationGenerator(t_set, o_set)
    population: List[ParseTree] = generator.generate(population_size, max_depth, generation_method)

    # Do things with the population
    for k in population:
        print("Tree:\n{}".format(k))


if __name__ == '__main__':
    main()

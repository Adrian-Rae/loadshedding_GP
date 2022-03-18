import math
import random
from typing import List

from src.GPAtom import Terminal, Operator, Variable, ConstantRange, Constant
from src.GPFitnessFunction import FitnessFunction
from src.GPOperator import GeneticOperator
from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator
from src.GPSelector import Selector, SelectionMethod


def main():
    # Establish the seed - leave commented for pure random
    seed = 2
    random.seed(seed)

    # Global Properties
    population_size = 1
    max_depth = 3
    generation_method = PopulationGenerator.Method.FULL
    selection_method = SelectionMethod.TOURNAMENT

    # Establish terminals and Operators
    x: Variable = Variable("x")
    alpha: ConstantRange = ConstantRange(-1, 1)
    one: Constant = Constant(1)

    mult: Operator = Operator("*", lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", lambda a, b: a + b, rep="({} + {})")
    minus: Operator = Operator("-", lambda a, b: a - b, rep="({} - {})")
    floor: Operator = Operator("floor", lambda a: math.floor(a))

    # Create a terminal and operator set
    t_set: List[Terminal] = [x, one, alpha]
    o_set: List[Operator] = [mult, minus]

    # Create a population
    generator: PopulationGenerator = PopulationGenerator(t_set, o_set)
    population: List[ParseTree] = generator.generate(population_size, max_depth, generation_method)

    # bind fitness cases to the function
    fit = FitnessFunction()
    cases = [
        ({"x": 1}, 0),
        ({"x": 2}, 3),
        ({"x": 3}, 8),
        ({"x": 4}, 15),
        ({"x": 5}, 24)
    ]
    for args, target in cases:
        fit.bind_case(args, target)

    print(*population)

    sel: Selector = Selector(fit, selection_method)
    op: GeneticOperator = GeneticOperator(sel, generator)
    print(op.mutation(population))


if __name__ == '__main__':
    main()

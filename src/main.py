import math
from typing import List

from src.GPAtom import Terminal, Operator, Variable, ConstantRange, Constant
from src.GPControlModel import GenerationalControlModel
from src.GPFitnessFunction import FitnessFunction
from src.GPGeneticOperator import GeneticOperatorSet, GeneticOperatorType
from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator
from src.GPSelector import SelectionMethod, Selector


def main():
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
    o_set: List[Operator] = [mult, minus, add, floor]

    # Create a control model
    model: GenerationalControlModel = GenerationalControlModel(
        population_size=100,
        max_tree_depth=5,
        terminal_set=t_set,
        operator_set=o_set,
        iteration_threshold=100,
        generation_method=PopulationGenerator.Method.FULL,
        selection_method=SelectionMethod.TOURNAMENT,
        explicit_convergence_condition=lambda max_fitness: max_fitness > 0.95,
        genetic_operators=[
            (GeneticOperatorType.CROSSOVER, 4),
            (GeneticOperatorType.MUTATION, 2),
            (GeneticOperatorType.PERMUTATION, 2),
            (GeneticOperatorType.EDITING, 2),
            (GeneticOperatorType.INVERSION, 4),
            (GeneticOperatorType.HOIST, 3)
        ],
        seed=11
    )

    cases = [
        ({"x": 1}, 0),
        ({"x": 2}, 3),
        ({"x": 2.5}, 5.25),
        ({"x": 3}, 8),
        ({"x": 4}, 15),
        ({"x": 5}, 24),
        ({"x": 6}, 35)
    ]
    for args, target in cases:
        model.bind_fitness_case(target, **args)

    model.evolve(
            action_on_evaluation=lambda iteration, optimal_member, best, avg: print("Iteration {}: [fitness {} / Avg. {}]: {}".format(iteration, best, avg, optimal_member)),
            action_on_converged=lambda best, fit: print("THIS IS THE BEST [Fitness {}]: {}".format(fit, best))
    )


if __name__ == '__main__':
    main()

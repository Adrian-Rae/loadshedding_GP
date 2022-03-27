import math
import random
from typing import List

from GPAtom import Terminal, Operator, Variable, ConstantRange, Constant
from GPControlModel import GenerationalControlModel
from GPGeneticOperator import GeneticOperatorType
from GPParseTree import ParseTree
from GPPopulationGenerator import PopulationGenerator
from GPSelector import SelectionMethod
from LoadSheddingData import DatasetManager


def main():
    # no of stages, including no load shedding
    nstages = 9
    total_population_size = 5

    seed = 1
    # Setup seed
    if seed is None:
        seed = random.randint(0, 100)
    random.seed(seed)

    # Establish terminals and Operators
    t: Variable = Variable("t")  # the timestamp
    s: Variable = Variable("s")  # the claimed stage
    var_std = [s, t]

    alpha: ConstantRange = ConstantRange(-2, 2)
    econst = [alpha]

    one: Constant = Constant(1)
    pi: Constant = Constant(math.pi, "π")
    const = [one, pi]

    mult: Operator = Operator("*", lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", lambda a, b: a + b, rep="{} + {}")
    op_std = [mult, add]

    divs: Operator = Operator("divs", lambda a, b: a / b if b != 0 else 1)
    sub: Operator = Operator("-", lambda a, b: a - b, rep="{} - {}")
    op_inv = [divs, sub]

    sine: Operator = Operator("sin", lambda a: math.sin(a))
    logs: Operator = Operator("logs", lambda a: math.log(a) if a > 0 else 0)
    floor: Operator = Operator("floor", lambda a: math.floor(a), rep=" ⌊{}⌋")
    op_ext = [sine, logs, floor]

    # Create a terminal and operator set
    t_set: List[Terminal] = var_std + econst + const
    o_set: List[Operator] = op_std + op_inv + op_ext

    LSManager = DatasetManager(nstages,mtype=DatasetManager.ModelType.EXTENDED)

    reducer = LSManager.get_reducer()

    cases = LSManager.generate_fitness_cases()
    # model_options = {
    #     "population_size": total_population_size,
    #     "max_tree_depth": 15,
    #     "terminal_set": t_set,
    #     "operator_set": o_set,
    #     "iteration_threshold": 2,
    #     "generation_method": PopulationGenerator.Method.FULL,
    #     "selection_proportion": 0.3,
    #     "selection_method": SelectionMethod.TOURNAMENT,
    #     "explicit_convergence_condition": lambda max_fitness: max_fitness > 0.95,
    #     "genetic_operators": [
    #         (GeneticOperatorType.CROSSOVER, 4),
    #         (GeneticOperatorType.MUTATION, 3),
    #         (GeneticOperatorType.PERMUTATION, 2),
    #         (GeneticOperatorType.EDITING, 2),
    #         (GeneticOperatorType.INVERSION, 2),
    #         (GeneticOperatorType.HOIST, 1)
    #     ],
    #     "fair_node_selection": False,
    #     "seed": 1,
    #     "error_aggregator": lambda S: sum(S) / len(S),
    #     "error_metric": lambda y, tar: abs(reducer(y) - tar) ** 2,
    #     "print_init": True,
    #     "parallelization": 5,
    #     "allow_trivial_exp": False
    # }
    #
    # # Create a control model
    # model: GenerationalControlModel = GenerationalControlModel(**model_options)
    # for args, target in cases:
    #     model.bind_fitness_case(target, **args)
    #
    # winner: ParseTree
    # winner, final_population = model.evolve(
    #     action_on_evaluation=lambda iteration, optimal_member, best, avg: print(
    #         "Iteration {}: [fitness {} / Avg. {}]: {}".format(iteration, best, avg, optimal_member)),
    #     action_on_converged=lambda best, fit: print("THIS IS THE BEST [Fitness {}]: {}".format(fit, best))
    # )
    #
    # # expect to see 2
    # classification, rlist = winner.classify(reducer, "s", [("t", 1423053934)], max_classes=nstages)
    # print(classification)
    # print(rlist)
    #
    # with open('../modeldata/model-{}.pkl'.format(seed), 'xb') as persist:
    #     winner.set_config(model_options)
    #     dill.dump(winner, persist)
    #
    # with open('../modeldata/model-{}.pkl'.format(seed), 'rb') as loading:
    #     revisited = dill.load(loading)
    #     print("Revisited", revisited)



if __name__ == '__main__':
    main()

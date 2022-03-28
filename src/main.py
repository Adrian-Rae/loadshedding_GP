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

SAVE_SRC = "../modeldata/seed_run_data.txt"


def main(seed):
    # ===============================================
    # GLOBAL PARAMETERS
    # ===============================================

    # Number of LS stages, including the 0 - NULL stage
    number_load_shedding_stages = 9

    # Number of members to generate in the population
    number_initial_population = 50

    # Max Depth of Individuals
    number_nested_expressions = 15

    # Number of iterations to bound iterative refinement
    number_max_iterations = 20

    # Parallelization
    number_concurrent_threads = 5

    # ===============================================
    # RANDOMNESS AND SEEDS
    # ===============================================

    # Setup seed
    if seed is None:
        seed = random.randint(0, 100)
    random.seed(seed)

    # ===============================================
    # DATASET LOADING, MANAGEMENT
    # ===============================================

    # Create the manager
    dataset_manager = DatasetManager(
        number_load_shedding_stages,
        # The type of model being used for load-shedding prediction. %SEE DOCUMENTATION
        mtype=DatasetManager.ModelType.EXTENDED
    )

    # Get a reduction function R from the manager
    reduction_functor = dataset_manager.get_reducer()

    # ===============================================
    # TERMINAL SET
    # ===============================================

    # Get variables based on the model
    var_std = dataset_manager.generate_variables()
    # Does something of the following, depending on the choice of model
    # t: Variable = Variable("t")  # the timestamp
    # s: Variable = Variable("s")  # the claimed stage
    # p: Variable = Variable("p")  # the electrical output
    # var_std = [s, t, p]

    # Ephemeral Constants
    alpha: ConstantRange = ConstantRange(-2, 2)
    econst = [alpha]

    # True Constants
    one: Constant = Constant(1)
    pi: Constant = Constant(math.pi, "pi")
    const = [one, pi]

    # Terminal Set
    t_set: List[Terminal] = var_std + econst + const

    # ===============================================
    # FUNCTION SET
    # ===============================================

    # Standard Operators
    mult: Operator = Operator("*", lambda a, b: a * b, rep="({} * {})")
    add: Operator = Operator("+", lambda a, b: a + b, rep="{} + {}")
    op_std = [mult, add]

    # Inverses
    divs: Operator = Operator("divs", lambda a, b: a / b if b != 0 else 1)
    sub: Operator = Operator("-", lambda a, b: a - b, rep="{} - {}")
    op_inv = [divs, sub]

    # Extended Operators
    sine: Operator = Operator("sin", lambda a: math.sin(a))
    logs: Operator = Operator("logs", lambda a: math.log(a) if a > 0 else 0)
    floor: Operator = Operator("floor", lambda a: math.floor(a))
    op_ext = [sine, logs, floor]

    # Operator (Function) Set
    o_set: List[Operator] = op_std + op_inv + op_ext

    # ===============================================
    # GENERATING FITNESS CASES - TRAINING
    # ===============================================

    fitness_cases = dataset_manager.generate_fitness_cases(insertion_factor=5)

    # partition into training and testing set
    random.shuffle(fitness_cases)
    half_index = len(fitness_cases) // 2
    fitness_cases_training = fitness_cases[:half_index]
    fitness_cases_testing = fitness_cases[half_index:]

    # ===============================================
    # ESTABLISHING CONTROL MODEL
    # ===============================================

    model_options = {
        "population_size": number_initial_population,
        "max_tree_depth": number_nested_expressions,
        "terminal_set": t_set,
        "operator_set": o_set,
        "iteration_threshold": number_max_iterations,
        "generation_method": PopulationGenerator.Method.GROW,
        "selection_proportion": 0.3,
        "selection_method": SelectionMethod.TOURNAMENT,
        "explicit_convergence_condition": lambda max_fitness: max_fitness > 0.95,
        "genetic_operators": [
            (GeneticOperatorType.CROSSOVER, 4),
            (GeneticOperatorType.MUTATION, 3),
            (GeneticOperatorType.PERMUTATION, 2),
            (GeneticOperatorType.EDITING, 2),
            (GeneticOperatorType.INVERSION, 2),
            (GeneticOperatorType.HOIST, 1)
        ],
        "fair_node_selection": False,
        "seed": seed,
        "error_aggregator": lambda S: sum(S) / len(S),
        "error_metric": lambda y, tar: abs(reduction_functor(y) - tar) ** 2,
        "print_init": True,
        "parallelization": number_concurrent_threads,
        "allow_trivial_exp": False
    }

    # Establish model using given parameters
    control_model: GenerationalControlModel = GenerationalControlModel(**model_options)

    # ===============================================
    # BIND FITNESS CASES TO CONTROL MODEL - TRAINING
    # ===============================================
    for args, target in fitness_cases_training:
        control_model.bind_fitness_case(target, **args)

    # ===============================================
    # MODEL EVOLUTION - TRAINING
    # ===============================================

    evolution_results = control_model.evolve(
        action_on_evaluation=lambda iteration, optimal_member, best, avg:
        print("Iteration {}: [Training Set Accuracy {} / Avg. {}]: {}".format(iteration, best, avg, optimal_member)),
    )

    for key in evolution_results.keys():
        value = evolution_results.get(key)
        print("{:>30} | {:<30}".format(key, str(value)))

    # Get the best individual
    winner = evolution_results.get("best_individual")
    train_accuracy = evolution_results.get("best_fitness")

    # evaluate the individual using the test set
    test_accuracy = control_model.evaluate_test_set(fitness_cases_testing, winner)
    print("{:>30} | {:<30}".format("Training Set Accuracy", train_accuracy))
    print("{:>30} | {:<30}".format("Testing Set Accuracy", test_accuracy))

    with open(SAVE_SRC, "a") as f:
        f.write("[SEED: {}, A_TRAIN: {}, A_TEST: {}]\n".format(seed, train_accuracy, test_accuracy))


if __name__ == '__main__':
    print("THIS OPERATION IS THREADED WITH DAEMON PROCESSES: DO NOT ATTEMPT A KEYBOARD INTERRUPT.")
    # with open(SAVE_SRC, "w"):
    #     pass
    for in_seed in range(2,11):
        print("STARTING SEED {}".format(in_seed))
        main(in_seed)

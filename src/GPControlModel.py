import random
import threading
from typing import List, Callable, Any, Tuple

import numpy as np
from numpy import argmax

from GPAtom import Terminal, Operator
from GPFitnessFunction import FitnessObjective, FitnessFunction, FitnessMeasure
from GPGeneticOperator import GeneticOperatorSet, GeneticOperatorType
from GPParseTree import ParseTree
from GPPopulationGenerator import PopulationGenerator
from GPSelector import SelectionMethod, Selector


class ControlModel:

    def __init__(
            self,
            population_size: int,
            max_tree_depth: int,
            terminal_set: List[Terminal],
            operator_set: List[Operator],
            generation_method: PopulationGenerator.Method = PopulationGenerator.Method.GROW,
            fair_node_selection: bool = True,
            selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
            selection_proportion: float = 0.3,
            seed: int = None,
            fitness_objective: FitnessObjective = FitnessObjective.MINIMISE,
            error_aggregator: Callable = lambda S: sum(S),
            error_metric: Callable = lambda y, t: abs(y - t),
            maximising_max_fitness: int = 1000000,
            allow_trivial_exp: bool = False,
            equality_threshold: float = 0.05,
            iteration_threshold: int = 100,
            explicit_convergence_condition: Callable[[float], bool] = lambda max_fitness: False,
            genetic_operators: List[Tuple[GeneticOperatorType, int]] = None,
            print_init: bool = True,
            simplify_final: bool = True,
            parallelization: int = 1
    ):
        self._population_size = population_size
        self._max_tree_depth = max_tree_depth
        self._terminal_set = terminal_set
        self._operator_set = operator_set
        self._generation_method = generation_method
        self._selection_method = selection_method
        self._selection_proportion = selection_proportion
        self._fitness_objective = fitness_objective
        self._error_aggregator = error_aggregator
        self._error_metric = error_metric
        self._maximising_max_fitness = maximising_max_fitness
        self._equality_threshold = equality_threshold
        self._allow_trivial_exp = allow_trivial_exp
        self._iteration_threshold = iteration_threshold
        self._explicit_convergence_condition = explicit_convergence_condition
        self._genetic_operators = []
        self._genetic_operator_weights = []
        self._simplify_final = simplify_final
        self._seed = seed
        self._parallelization = parallelization
        self._lock = threading.Lock()

        self._population_buffer = []

        # State of evolution - whether the model has begun processing
        self._iteration = 0
        self._in_progress = False

        # Setup the initial population generator
        self._population_generator: PopulationGenerator = PopulationGenerator(self._terminal_set, self._operator_set, parallelization=self._parallelization)

        # Setup the fitness function
        self._fitness_function: FitnessFunction = FitnessFunction(
            self._fitness_objective,
            self._error_aggregator,
            self._error_metric,
            self._maximising_max_fitness,
            self._equality_threshold,
            self._allow_trivial_exp
        )

        # Setup Selection and Genetic Operator mechanisms
        self._genetic_selection: Selector = Selector(
            self._fitness_function,
            self._selection_method,
            self._selection_proportion
        )
        self._genetic_operator_set: GeneticOperatorSet = GeneticOperatorSet(self._genetic_selection,
                                                                            self._population_generator)

        # Load operators
        if genetic_operators is not None:
            for op, weight in genetic_operators:
                self._genetic_operators.append(op)
                self._genetic_operator_weights.append(weight)
        else:
            for op, weight in GeneticOperatorSet.default_set:
                self._genetic_operators.append(op)
                self._genetic_operator_weights.append(weight)

        # Create the initial population
        self._population: List[ParseTree] = self._population_generator.generate(
            self._population_size,
            self._max_tree_depth,
            self._generation_method,
            force_fairness=fair_node_selection
        )

        # Optimal fitness
        self._optimal_fitness = None
        self._optimal_member = None
        self._avg_fitness = None

        # Print config
        if print_init:
            print(
                "{}".format(63 * "="),
                "{:>30} | {:<30}".format("Generational Control Model",
                                         "{} iteration bound".format(self._iteration_threshold)),
                "{:>30} | {:<30}".format("Seed", seed),
                "{:>30} | {:<30}".format("Initial Pop. Size", self._population_size),
                "{:>30} | {:<30}".format("Max Tree Depth", self._max_tree_depth),
                "{:>30} | {:<30}".format("Pop. Generation Method", self._generation_method),
                "{:>30} | {:<30}".format("Selection Method", self._selection_method),
                "{:>30} | {:<30}".format("Equality Threshold", self._equality_threshold),
                "{}".format(63 * "="),
                sep="\n",
                end="\n"
            )

    def get_seed(self):
        return self._seed

    def bind_fitness_case(self, target, **kwargs):
        if not self._in_progress:
            self._fitness_function.bind_case(kwargs, target)
        else:
            raise InvalidOperationStateException

    def next(
            self,
            action_on_evaluation: Callable[[int, ParseTree, float, float], Any] = lambda iteration, optimal_member,
                                                                                         max_fitness, avg_fitness: None,
            action_on_converged: Callable[[ParseTree, int], Any] = lambda optimal_member, fitness: None
    ):
        # begin
        self._in_progress = True

        # evolutionary action
        self._survive()

        # evaluate changes made to new population
        self._evaluate()

        # optional action taken on population following iteration
        action_on_evaluation(self._iteration, self._optimal_member, self._optimal_fitness, self._avg_fitness)

        return not self._converged(action_on_converged=action_on_converged)

    def evolve(
            self,
            action_on_evaluation: Callable[[int, ParseTree, float, float], Any] = lambda iteration, optimal_member,
                                                                                         max_fitness, avg_fitness: None,
            action_on_converged: Callable[[ParseTree, int], Any] = lambda optimal_member, fitness: None
    ):

        while self.next(
                action_on_evaluation=action_on_evaluation,
                action_on_converged=action_on_converged
        ):
            pass

        # Return the optimal member
        return self._optimal_member, self._population

    def _evaluate(self):

        # batch process adjusted values first so as to not calculate every time
        self._fitness_function.predefine_aggregate(self._population)

        fitness_normalised = [
            self._fitness_function.fitness(
                individual=ind,
                measure=FitnessMeasure.NORMALIZED
            )
            for ind in self._population
        ]
        maximizing_index = argmax(fitness_normalised)
        self._optimal_member = self._population[maximizing_index]
        self._optimal_fitness = self._fitness_function.fitness(self._optimal_member, measure=FitnessMeasure.ADJUSTED)
        self._avg_fitness = self._fitness_function.average()

    def optimal(self):
        return self._optimal_member

    def _simplify(self):
        # simplify the optimal member
        identity = str(self._optimal_member)
        while True:
            pre_simplify = identity
            self._optimal_member = self._genetic_operator_set.editing([self._optimal_member])[0]
            identity = str(self._optimal_member)
            if pre_simplify == identity:
                break

    def _converged(self, action_on_converged: Callable[[ParseTree, int], Any] = lambda optimal_member, fitness: None):
        self._iteration += 1
        iterations_reached = self._iteration >= self._iteration_threshold
        explicit_convergence_achieved = self._explicit_convergence_condition(self._optimal_fitness)
        convergence_met = iterations_reached or explicit_convergence_achieved
        if convergence_met:
            # simplify the final result
            self._simplify()
            # do whatever is specified
            self._on_converged(action_on_converged=action_on_converged)
        return convergence_met

    def _on_converged(self,
                      action_on_converged: Callable[[ParseTree, int], Any] = lambda optimal_member, fitness: None):
        action_on_converged(self._optimal_member, self._optimal_fitness)

    def _survive(self):
        pass


class GenerationalControlModel(ControlModel):

    def _survive(self):
        super()._survive()

        # shuffle the population
        random.shuffle(self._population)

        threads = []
        # Parallelize based on factor
        for pop_subset in np.array_split(self._population, self._parallelization):
            operative_thread = threading.Thread(target=self._concurrent_survive, args=(pop_subset,),
                                                daemon=True)
            operative_thread.start()
            threads.append(operative_thread)

        # wait for threads to finish
        for thread in threads:
            thread.join()

        # return the new population
        self._population = self._population_buffer

        # empty the buffer
        self._population_buffer = []

        # Serial Execution
        # while len(new_population) < self._population_size:
        #
        #     # Choose an operator
        #     genetic_operator = random.choices(self._genetic_operators, weights=self._genetic_operator_weights, k=1)[0]
        #     new_subset = self._genetic_operator_set.operate(self._population, genetic_operator)
        #
        #     for child in new_subset:
        #         if len(new_population) >= self._population_size:
        #             break
        #         new_population.append(child)
        #
        # self._population = new_population

    def _concurrent_survive(self, sub_population):
        # while there is more to add to the next population
        new_population: List[ParseTree] = []

        while len(new_population) < len(sub_population):

            # Choose an operator
            genetic_operator = random.choices(self._genetic_operators, weights=self._genetic_operator_weights, k=1)[0]
            new_subset = self._genetic_operator_set.operate(sub_population, genetic_operator)

            for child in new_subset:
                if len(new_population) >= len(sub_population):
                    break
                new_population.append(child)

        # acquire lock to write to buffer
        with self._lock:
            self._population_buffer += new_population


class SteadyStateModel(ControlModel):

    def _survive(self):
        super()._survive()


class InvalidOperationStateException(Exception):
    def __init__(self):
        super().__init__("Cannot affect the generational control parameters with evolution in progress.")

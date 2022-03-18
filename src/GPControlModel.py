import random
from typing import List, Callable, Any, Tuple

from numpy import argmax

from src.GPAtom import Terminal, Operator
from src.GPFitnessFunction import FitnessObjective, FitnessFunction, FitnessMeasure
from src.GPGeneticOperator import GeneticOperatorSet, GeneticOperatorType
from src.GPParseTree import ParseTree
from src.GPPopulationGenerator import PopulationGenerator
from src.GPSelector import SelectionMethod, Selector


class ControlModel:
    def __init__(
            self,
            population_size: int,
            max_tree_depth: int,
            terminal_set: List[Terminal],
            operator_set: List[Operator],
            generation_method: PopulationGenerator.Method = PopulationGenerator.Method.GROW,
            selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
            seed: int = None,
            fitness_objective: FitnessObjective = FitnessObjective.MINIMISE,
            error_aggregator: Callable = lambda S: sum(S),
            error_metric: Callable = lambda y, t: abs(y - t),
            maximising_max_fitness: int = 1000000,
            equality_threshold: float = 0.05,
            iteration_threshold: int = 500,
            explicit_convergence_condition: Callable[[float], bool] = lambda max_fitness: False,
            genetic_operators=GeneticOperatorSet.default_set
    ):
        self._population_size = population_size
        self._max_tree_depth = max_tree_depth
        self._terminal_set = terminal_set
        self._operator_set = operator_set
        self._generation_method = generation_method
        self._selection_method = selection_method
        self._fitness_objective = fitness_objective
        self._error_aggregator = error_aggregator
        self._error_metric = error_metric
        self._maximising_max_fitness = maximising_max_fitness
        self._equality_threshold = equality_threshold
        self._iteration_threshold = iteration_threshold
        self._explicit_convergence_condition = explicit_convergence_condition
        self._genetic_operators = genetic_operators

        # Setup seed
        if seed is not None:
            random.seed(seed)

        # State of evolution - whether the model has begun processing
        self._iteration = 0
        self._in_progress = False

        # Setup the initial population generator
        self._population_generator: PopulationGenerator = PopulationGenerator(self._terminal_set, self._operator_set)

        # Setup the fitness function
        self._fitness_function: FitnessFunction = FitnessFunction(
            self._fitness_objective,
            self._error_aggregator,
            self._error_metric,
            self._maximising_max_fitness,
            self._equality_threshold
        )

        # Setup Selection and Genetic Operator mechanisms
        self._genetic_selection: Selector = Selector(self._fitness_function, self._selection_method)
        self._genetic_operator_set: GeneticOperatorSet = GeneticOperatorSet(self._genetic_selection,
                                                                            self._population_generator)

        # Create the initial population
        self._population: List[ParseTree] = self._population_generator.generate(
            self._population_size,
            self._max_tree_depth,
            self._generation_method
        )

        # Optimal fitness
        self._optimal_fitness = None
        self._optimal_member = None
        self._avg_fitness = None

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

    def _converged(self, action_on_converged: Callable[[ParseTree, int], Any] = lambda optimal_member, fitness: None):
        self._iteration += 1
        iterations_reached = self._iteration >= self._iteration_threshold
        explicit_convergence_achieved = self._explicit_convergence_condition(self._optimal_fitness)
        convergence_met = iterations_reached or explicit_convergence_achieved
        if convergence_met:
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

        # while there is more to add to the next population
        new_population: List[ParseTree] = []
        while len(new_population) < self._population_size:

            # Choose an operator
            genetic_operator = random.choice(self._genetic_operators)
            new_subset = self._genetic_operator_set.operate(self._population, genetic_operator)

            for child in new_subset:
                if len(new_population) >= self._population_size:
                    break
                new_population.append(child)

        self._population = new_population


class SteadyStateModel(ControlModel):

    def _survive(self):
        super()._survive()


class InvalidOperationStateException(Exception):
    def __init__(self):
        super().__init__("Cannot affect the generational control parameters with evolution in progress.")

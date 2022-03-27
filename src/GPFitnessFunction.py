import math
from collections import Callable
from enum import Enum
from typing import Dict, List

from GPParseTree import ParseTree


class FitnessObjective(Enum):
    MINIMISE = 0,
    MAXIMISE = 1


class FitnessMeasure(Enum):
    RAW = 0
    STANDARDIZED = 1
    ADJUSTED = 2
    NORMALIZED = 3
    HITS_RATIO = 4


class FitnessFunction:

    def __init__(self,
                 objective: FitnessObjective = FitnessObjective.MINIMISE,
                 aggregator: Callable = lambda S: sum(S),
                 error: Callable = lambda y, t: abs(y - t),
                 max_fitness: int = 1000000,
                 equality_bound: float = 0.05,
                 allow_trivial: bool = False,
                 ) -> None:
        self._objective = objective
        self._fitness_cases = []
        self._aggregator = aggregator
        self._error = error
        self._method_index = [
            self._raw_fitness,
            self._standardized_fitness,
            self._adjusted_fitness,
            self._normalized_fitness,
            self._hits_ratio
        ]
        self._fitness_max = max_fitness
        self._bound = equality_bound
        self._aggregate_adjusted = None
        self._aggregate_members = None
        self._allow_trivial = allow_trivial

    def bind_case(self, args: Dict, target) -> None:
        self._fitness_cases.append((args, target))

    def _eval(self, individual: ParseTree) -> float:
        accumulator = []
        for args, target in self._fitness_cases:
            y = individual.eval(**args)

            # do not allow trivial expression
            is_trivial = (not individual.get_root().is_parameterized()) and not self._allow_trivial

            e = self._error(y, target) + ( (self._fitness_max if self._objective is FitnessObjective.MINIMISE else -self._fitness_max) if is_trivial else 0)
            accumulator.append(e)
        return self._aggregator(accumulator)

    def fitness(self, individual: ParseTree, population=None, measure: FitnessMeasure = FitnessMeasure.RAW) -> float:
        index = measure.value
        method = self._method_index[index]
        return method(individual=individual, population=population)

    def _raw_fitness(self, individual: ParseTree, **kwargs) -> float:
        return self._eval(individual)

    def _standardized_fitness(self, individual: ParseTree, **kwargs) -> float:
        return self._raw_fitness(individual) \
            if self._objective is FitnessObjective.MINIMISE \
            else self._fitness_max - self._raw_fitness(individual)

    def _adjusted_fitness(self, individual: ParseTree, **kwargs) -> float:
        return 1 / (1 + self._standardized_fitness(individual))

    def predefine_aggregate(self, population: List[ParseTree]):
        self._aggregate_members = len(population)
        self._aggregate_adjusted = sum([self._adjusted_fitness(p) for p in population])

    def average(self, population: List[ParseTree] = None):
        factor = self._aggregate_adjusted if population is None else sum(
            [self._adjusted_fitness(p) for p in population])
        if factor is None:
            raise InvalidAggregationException
        no_mem = self._aggregate_members if population is None else len(population)
        if no_mem is None:
            raise InvalidAggregationException
        return factor / no_mem

    def _normalized_fitness(self, individual: ParseTree, population: List[ParseTree], **kwargs) -> float:
        factor = self._aggregate_adjusted if population is None else sum(
            [self._adjusted_fitness(p) for p in population])
        if factor is None:
            raise InvalidAggregationException
        return self._adjusted_fitness(individual) / factor

    def _hits_ratio(self, individual: ParseTree, **kwargs) -> float:
        return len([
            None for args, target in self._fitness_cases
            # if abs(individual.eval(**args) - target) < self._bound
            if math.isclose(individual.eval(**args), target, rel_tol=self._bound)
        ]) / len(self._fitness_cases)


class InvalidAggregationException(Exception):
    def __init__(self):
        super().__init__(
            "No pre-existing aggregate adjusted value for this population exists. Please specify the population when "
            "calculating the normalized fitness, or make use of the predefine_aggregate() function.")

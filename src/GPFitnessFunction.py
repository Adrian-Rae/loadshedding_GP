import sys
from collections import Callable
from enum import Enum
from typing import Dict, List

from src.GPParseTree import ParseTree


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
                 equality_bound: float = 0.05
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

    def bind_case(self, args: Dict, target) -> None:
        self._fitness_cases.append((args, target))

    def _eval(self, individual: ParseTree) -> float:
        accumulator = []
        for args, target in self._fitness_cases:
            y = individual.eval(**args)
            e = self._error(y=y, t=target)
            accumulator.append(e)
        return self._aggregator(accumulator)

    def fitness(self, individual: ParseTree, population=None, measure: FitnessMeasure = FitnessMeasure.RAW) -> float:
        if population is None:
            population = [individual]
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

    def _normalized_fitness(self, individual: ParseTree, population: List[ParseTree], **kwargs) -> float:
        return self._adjusted_fitness(individual) / sum([self._adjusted_fitness(p) for p in population])

    def _hits_ratio(self, individual: ParseTree, **kwargs) -> float:
        return len([
            None for args, target in self._fitness_cases
            if abs(individual.eval(**args) - target) < self._bound
        ]) / len(self._fitness_cases)

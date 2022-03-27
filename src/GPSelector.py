import random
from enum import Enum
from math import ceil
from typing import List

import numpy as np

from GPFitnessFunction import FitnessFunction, FitnessMeasure
from GPParseTree import ParseTree


class SelectionMethod(Enum):
    FITNESS_PROPORTIONATE = 0
    TOURNAMENT = 1


class Selector:

    def __init__(self, fitness: FitnessFunction, method: SelectionMethod = SelectionMethod.FITNESS_PROPORTIONATE, proportion: float = 0.3) -> None:
        self._fitness = fitness
        self._method_index = [
            self._select_proportionate,
            self._select_tournament
        ]
        self._method = method
        self._proportion = proportion

    def select(self, population: List[ParseTree]) -> ParseTree:
        return self._method_index[self._method.value](population=population, proportion=self._proportion)

    def _select_proportionate(self, population: List[ParseTree], **kwargs) -> ParseTree:

        # predetermine aggregate for performance
        self._fitness.predefine_aggregate(population)

        nf = [self._fitness.fitness(
            k,
            measure=FitnessMeasure.NORMALIZED,
        ) for k in population]

        # no. of occurrences for each individual in a pool prone to rounding errors - use cumulative threshold
        # no = [n_pop * f for f in nf]

        # instead, make a cumulative frequency array
        cf = nf.copy()
        for i in range(1, len(cf)):
            cf[i] += cf[i - 1]

        # choose a random number and select the individual corresponding to the bin the number falls in
        r = random.random()

        for i in range(len(cf)):
            if r < cf[i]:
                return population[i]

    def _select_tournament(self, population: List[ParseTree], proportion: float = 0.3) -> ParseTree:

        if not ((0 < proportion) and (proportion <= 1)):
            raise InvalidTournamentProportionException

        nt = ceil(proportion * len(population))
        subpopulation = random.choices(population, k=nt)
        return subpopulation[
            np.argmax([
                self._fitness.fitness(s, measure=FitnessMeasure.ADJUSTED)
                for s in subpopulation
            ])]


class InvalidTournamentProportionException(Exception):
    def __init__(self):
        super().__init__(
            "An invalid tournament proportion was passed. Valid population proportions must be in the range (0,1]."
        )

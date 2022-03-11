import random
from enum import Enum
from typing import List

import numpy as np

from src.GPFitnessFunction import FitnessFunction, FitnessMeasure
from src.GPParseTree import ParseTree


class SelectionMethod(Enum):
    FITNESS_PROPORTIONATE = 0
    TOURNAMENT = 1


class Selector:

    def __init__(self, population: List[ParseTree], fitness: FitnessFunction) -> None:
        self._population = population
        self._fitness = fitness
        self._method_index = [
            self._select_proportionate,
            self._select_tournament
        ]
        self._default_proportion = 0.3

    def select(self, method: SelectionMethod = SelectionMethod.FITNESS_PROPORTIONATE) -> ParseTree:
        return self._method_index[method.value]()

    def _select_proportionate(self, **kwargs) -> ParseTree:
        n_pop = len(self._population)
        nf = [self._fitness.fitness(
            k,
            measure=FitnessMeasure.NORMALIZED,
            population=self._population
        ) for k in self._population]

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
                return self._population[i]

    def _select_tournament(self, proportion: float = None) -> ParseTree:
        proportion = self._default_proportion if proportion is None else proportion
        nt = round(proportion * len(self._population))
        subpopulation = random.choices(self._population, k=nt)
        return subpopulation[
            np.argmax([
                self._fitness.fitness(s, measure=FitnessMeasure.ADJUSTED)
                for s in subpopulation
            ])]

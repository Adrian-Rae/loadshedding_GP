import math
from enum import Enum


class ReductionFunction(Enum):
    class _Maps:
        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        def sigmoid_inv(x: float) -> float:
            return math.log(x) - math.log(1 - x)

        def unit_arctan(x: float) -> float:
            return 2 * math.atan(x) / math.pi

        def unit_tan(x: float) -> float:
            return math.tan(math.pi * x / 2)

        def unit_tanh(x: float) -> float:
            return (1 + math.tanh(x)) / 2

        def unit_arctanh(x: float) -> float:
            return math.atanh(2 * x - 1)

    SIGMOID = (_Maps.sigmoid, _Maps.sigmoid_inv)
    ARCTAN = (_Maps.unit_arctan, _Maps.unit_tan)
    TANH = (_Maps.unit_tanh, _Maps.unit_arctanh)




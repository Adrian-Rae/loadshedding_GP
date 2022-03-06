import random
import sys
from typing import Callable


class Atom:
    """
    The most basic symbolic unit of an expression. Is an abstraction of a terminal or operator that can be evaluated.
    """

    def arity(self) -> int:
        """
        The number of dependencies needed in order to evaluate the atomic unit. Defaults to 0 as an instance of Atom
        is not used in practise.

        :return: The integer arity of the Atom.
        """

        return 0

    def instance(self):
        return self

    def eval(self, *args):
        self._validate_args(*args)
        return 0

    def _validate_args(self, *args):
        # Check here that number of formal and actual arguments match
        if not self.arity() == len(args):
            raise InvalidOperatorParameterBindingException(self.arity(), len(args), str(self))


class Terminal(Atom):
    """
    The most basic terminating unit of an expression. Can be directly evaluated without parameterization.
    """

    def arity(self) -> int:
        """
        The number of dependencies needed in order to evaluate the Terminal. Defaults to 0 as no arguments are
        needed to evaluate a terminating expression.

        :return: The integer arity of the Terminal.
        """

        return 0

    def instance(self):
        return self

    def eval(self, *args):
        self._validate_args(*args)
        return 0


class Variable(Terminal):

    def __init__(self, name: str, value=None):
        self._name = name
        self._value = value

    def bind(self, value):
        self._value = value

    def instance(self):
        return self

    def eval(self, *args):
        self._validate_args(*args)
        return self._value

    def __str__(self):
        return "{}[{}]".format(self._name, self.eval())


class Constant(Terminal):

    def __init__(self, value=None):
        self._value = value

    def eval(self, *args):
        self._validate_args(*args)
        return self._value

    def instance(self):
        return self

    def __str__(self):
        return "{}".format(str(self._value))


class ConstantRange(Terminal):

    def __init__(self, lower_bound, upper_bound):
        self._lower = lower_bound
        self._upper = upper_bound

    def eval(self, *args):
        self._validate_args(*args)
        return 0

    def instance(self):
        instance_value = self._lower + random.random() * (self._upper - self._lower)
        return Constant(instance_value)

    def __str__(self):
        return "<Unbound constant variable>"


class Operator(Atom):
    """
    The most basic non-terminating unit of an expression. Cannot be directly evaluated without parameterization.
    """

    def __init__(self, name: str, arity: int, evaluation_procedure: Callable = lambda: 0):
        """
        Constructor for an operator.

        :param name: String representation of the operator
        :param arity: The number of arguments needed to evaluate the expression defined by the use of the operator.
        :param evaluation_procedure: The resolution method used to evaluate an expression containing the operator.
        """

        self._name = name
        self._arity = arity
        self._evaluation_procedure = evaluation_procedure

    def arity(self) -> int:
        """
        The number of dependencies needed in order to evaluate the operation. Defaults to the arity specified during
        instantiation.

        :return: The integer arity of the Operator.
        """
        return self._arity

    def instance(self):
        return self

    def eval(self, *args):
        self._validate_args(*args)
        return self._evaluation_procedure(*args)

    def __str__(self):
        return self._name


# EXCEPTIONS

class InvalidOperatorParameterBindingException(Exception):
    def __init__(self, expected: int, actual: int, name: str = "<anon>"):
        super().__init__("Number of formal and actual arguments for the operator {} do not match (Expected {}, "
                         "Received {})".format(name, expected, actual))

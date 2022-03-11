import random
from inspect import signature
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

    def eval_str(self, *args) -> str:
        self._validate_args(*args)
        return str(self)

    def _validate_args(self, *args):
        # Check here that number of formal and actual arguments match
        if not self.arity() == len(args):
            raise InvalidOperatorParameterBindingException(self.arity(), len(args), str(self))

    def __str__(self):
        return "<anon>()"


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

    def __str__(self):
        return "<anon>"


class Variable(Terminal):

    def __init__(self, name: str, value=None):
        self._name = name
        self._value = value

    def bind(self, value):
        self._value = value

    def is_bound(self):
        return self._value is not None

    def instance(self):
        return self

    def eval(self, *args):
        self._validate_args(*args)
        return self._value

    def _validate_args(self, *args):
        if not self.is_bound():
            raise InvalidVariableBindingException(self._name)
        super()._validate_args(*args)

    def __str__(self):
        return "{}".format(self._name)


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
        return "<Unbound anon>"


class Operator(Atom):
    """
    The most basic non-terminating unit of an expression. Cannot be directly evaluated without parameterization.
    """

    def __init__(self, name: str, evaluation_procedure: Callable, rep: str = None):
        """
        Constructor for an operator.

        :param name: String representation of the operator
        :param evaluation_procedure: The resolution method used to evaluate an expression containing the operator.
        """

        self._name = name
        self._arity = len(signature(evaluation_procedure).parameters)

        self._evaluation_procedure = evaluation_procedure
        self._representation = rep

        if rep is None:
            self._representation = "{}({})".format(self._name, ",".join(["{}" for _ in range(self._arity)]))
        else:
            self._validate_rep(rep, self._arity)

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

    def eval_str(self, *args) -> str:
        self._validate_args(*args)
        return self._representation.format(*args)

    def _validate_rep(self, rep: str, arity: int):
        pars: int = rep.count("{}")
        if not pars == arity:
            raise InvalidOperatorRepresentationBindingException(arity, pars, self._name)

    def __str__(self):
        return self._name


# EXCEPTIONS

class InvalidVariableBindingException(Exception):
    def __init__(self, name: str = "<anon>"):
        super().__init__("The variable {} is not bound to a value. Use {}.bind(<value>)".format(name, name))


class InvalidOperatorParameterBindingException(Exception):
    def __init__(self, expected: int, actual: int, name: str = "<anon>"):
        super().__init__("Number of formal and actual arguments for the operator {} do not match (Expected {}, "
                         "Received {})".format(name, expected, actual))


class InvalidOperatorRepresentationBindingException(Exception):
    def __init__(self, expected: int, actual: int, name: str = "<anon>"):
        super().__init__("Representation of the operator {} does not match arity (Expected {}, "
                         "Received {})".format(name, expected, actual))

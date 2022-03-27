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

    def copy(self) -> 'Atom':
        """
        Create a deep copy of the target Atom.
        :return: An distinct Atom with identical properties to the current Atom.
        """
        return Atom()

    def is_parameterized(self) -> bool:
        """
        An indicator of whether the given Atom requires a future binding (such as that of a variable to a value) to
        be evaluated.
        :return: A boolean indicating whether or not the Atom requires binding in order to be evaluated.
        """
        return False

    def instance(self) -> 'Atom':
        """
        Allows for an instance of an item to be generated for when it is needed in an ADT. Used to ensure uniqueness
        across many sources, but a uniform presence in an alphabet list.
        :return: An instance of the desired Atom type
        """
        return self

    def eval(self, *args):
        """
        The concrete representation of an Atomic value.
        :param args: Arguments required to evaluate the concrete value of the Atom.
        :return: The concrete evaluation of the Atom.
        """
        self._validate_args(*args)
        return 0

    def eval_str(self, *args) -> str:
        """
        The symbolic representation of the Atom
        :param args: Arguments required to evaluate the symbolic value of the Atom.
        :return: The symbolic evaluation of the Atom.
        """
        self._validate_args_str(*args)
        return str(self)

    def _validate_args(self, *args) -> None:
        """
        Validate the arguments needed to evaluate a concrete representation of an Atomic value.
        :param args: Arguments required to evaluate the concrete value of the Atom.
        """
        # Check here that number of formal and actual arguments match
        if not self.arity() == len(args):
            raise InvalidOperatorParameterBindingException(self.arity(), len(args), str(self))

    def _validate_args_str(self, *args) -> None:
        """
        Validate the arguments needed to evaluate a symbolic representation of an Atomic value.
        :param args: Arguments required to evaluate the symbolic value of the Atom.
        """
        return self._validate_args(*args)

    def __str__(self) -> str:
        """
        The string representation of an Atom.
        :return: The string representation of an Atom.
        """
        return "<anon>()"


class Terminal(Atom):
    """
    The most basic terminating unit of an expression. Can be directly evaluated without parameterization.
    """

    def copy(self) -> 'Terminal':
        """
        Create a deep copy of the target Terminal.
        :return: An distinct Terminal with identical properties to the current Terminal.
        """
        return Terminal()


class Variable(Terminal):
    """
    A concrete terminating unit of an expression. Cannot be directly evaluated without parameterization.
    """

    def __init__(self, name: str, value=None) -> None:
        """
        Constructor for the Variable.
        :param name: Name of the Variable.
        :param value: (Optional) The initial value bound to the Variable.
        """
        super().__init__()
        self._name = name
        self._value = value

    def copy(self) -> 'Variable':
        """
        Create a deep copy of the target Variable.
        :return: An distinct Variable with identical properties to the current Variable.
        """
        return Variable(self._name, self._value)

    def bind(self, value) -> None:
        """
        Bind a value to the current Variable.
        :param value: The value to be bound to the Variable.
        """
        self._value = value

    def _is_bound(self) -> bool:
        """
        Indicates whether or not there is a value bound to the variable.
        :return: A boolean indicator of the binding state.
        """
        return self._value is not None

    def is_parameterized(self) -> bool:
        """
        An indicator of whether the given Atom requires a future binding. This is always true for a Variable.
        :return: A boolean indicating whether or not the Variable requires binding in order to be evaluated.
        """
        return True

    def eval(self, *args):
        self._validate_args(*args)
        return self._value

    def _validate_args(self, *args):
        if not self._is_bound():
            raise InvalidVariableBindingException(self._name)
        super()._validate_args(*args)

    def __str__(self):
        return "{}".format(self._name)


class Constant(Terminal):

    def __init__(self, value, name=None):
        super().__init__()
        self._value = value
        self._name = name if name is not None else "{:.2f}".format(self._value)

    def copy(self) -> 'Constant':
        return Constant(self._value, self._name)

    def eval(self, *args):
        self._validate_args(*args)
        return self._value

    def instance(self):
        return self

    def __str__(self):
        return "{}".format(self._name)


class ConstantRange(Terminal):

    def __init__(self, lower_bound, upper_bound, choose = None):
        super().__init__()
        self._lower = lower_bound
        self._upper = upper_bound
        self._choose = (lambda: self._lower + random.random() * (self._upper - self._lower)) if choose is None else choose

    def copy(self) -> 'ConstantRange':
        return ConstantRange(self._lower, self._upper)

    def eval(self, *args):
        self._validate_args(*args)
        return 0

    def instance(self):
        instance_value = self._choose()
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

        super().__init__()
        self._name = name
        self._arity = len(signature(evaluation_procedure).parameters)

        self._evaluation_procedure = evaluation_procedure
        self._representation = rep

        if rep is None:
            self._representation = "{}({})".format(self._name, ",".join(["{}" for _ in range(self._arity)]))
        else:
            self._validate_rep(rep, self._arity)

    def copy(self) -> 'Operator':
        return Operator(self._name, self._evaluation_procedure, self._representation)

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

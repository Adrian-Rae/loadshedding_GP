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


class Operator(Atom):
    """
    The most basic non-terminating unit of an expression. Cannot be directly evaluated without parameterization.
    """

    def __init__(self, arity: int, evaluation_procedure: Callable = lambda x: x):
        """
        Constructor for an operator.

        :param arity: The number of arguments needed to evaluate the expression defined by the use of the operator.
        :param evaluation_procedure: The resolution method used to evaluate an expression containing the operator.
        """

        self._arity = arity
        self._evaluation_procedure = evaluation_procedure

    def arity(self) -> int:
        """
        The number of dependencies needed in order to evaluate the operation. Defaults to the arity specified during
        instantiation.

        :return: The integer arity of the Operator.
        """
        return self._arity


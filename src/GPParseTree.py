import random

from typing import List

from GPAtom import *
from GPExceptions import InvalidDepthException, NodeTerminationException


class Node(Generic[T]):
    def __init__(self, value: Atom[T]) -> None:
        self._value: Atom = value
        self._children: List[Node] = []

    def add_child(self, child_value: 'Node[T]') -> None:
        if self.is_terminal():
            raise NodeTerminationException

        self._children.append(child_value)

    def arity(self):
        return self._value.arity()

    def is_terminal(self):
        return self.arity() == 0

    def get_children(self):
        return self._children

    def get_depth(self):
        if len(self._children) == 0:
            return 1
        return 1 + max([k.get_depth() for k in self._children])

    def __str__(self, level=0):
        ret = "\t" * level + repr(self._value) + str(self.get_depth()) + "\n"
        for child in self._children:
            ret += child.__str__(level + 1)
        return ret


class ParseTree(Generic[T]):
    __create_key = object()

    @classmethod
    def random(cls, max_depth: int, terminal_set: TerminalSetType, operator_set: OperatorSetType) -> 'ParseTree':
        # A tree cannot be trivial
        if max_depth < 2:
            raise InvalidDepthException

        # Generate a non-terminal root from one atom in the operator set
        root_atom: Operator[T] = random.choice(operator_set).instance()
        root: Node[T] = Node[T](root_atom)

        # fill the children according to the remaining levels
        cls.__fill_level(root, max_depth - 1, terminal_set, operator_set)

        return ParseTree(cls.__create_key, root)

    @classmethod
    # Given a node, fill children randomly given a specific number of levels and terminal and operator sets
    def __fill_level(cls, root: Node[T], rem_levels: int, terminal_set: TerminalSetType,
                     operator_set: OperatorSetType) -> None:

        if rem_levels == 0:
            return

        # the number of children for the node
        n_child: int = root.arity()

        # the selection set is both terminals and operators, unless there is only 1 remaining level, in which case it
        # is only terminals
        child_atom_options: UnionSetType = terminal_set if rem_levels < 2 else terminal_set + operator_set

        # for each argument/child
        for i in range(n_child):
            child_atom = random.choice(child_atom_options).instance()
            new_child: Node[T] = Node[T](child_atom)
            root.add_child(new_child)
            cls.__fill_level(new_child, rem_levels - 1, terminal_set, operator_set)

    # Private Constructor
    def __init__(self, key: object, root: Node[T]) -> None:
        assert (key == ParseTree.__create_key), "Make use of the ParseTree.Random method to create a new, random tree."
        self._root = root

    def get_depth(self):
        return self._root.get_depth()

    def __str__(self):
        return str(self._root)

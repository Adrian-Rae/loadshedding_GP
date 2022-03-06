import random
from math import log, floor
from typing import List, cast
from GPAtom import *


class ParseTree:
    """
    Class which defines a hierarchy of operations in a tree format.
    """

    # Creation key: used to specify the uniqueness of instance objects for determining class methods.
    __create_key = object()

    class Node:
        """
        Nested class which encapsulates a node in the Parse Tree.
        """

        def __init__(self, value: Atom) -> None:
            """
            Constructor for a Node object.

            :param value: An Atom which encapsulates the node's value.
            """

            self._value: Atom = value
            self._children: List[ParseTree.Node] = []

        def eval(self, **kwargs):
            # Evaluate the inner expression given the children

            # If Terminal
            if self.is_terminal():

                # See if it matches one of the keyword args and bind to the Variable
                arg: str = str(self._value)
                search = kwargs.get(arg)
                if search is not None:
                    cast(Variable, self._value).bind(search)

                return self._value.eval()

            # Else, evaluate all children first
            return self._value.eval(*(child.eval(**kwargs) for child in self._children))

        def eval_str(self) -> str:
            # Evaluate the inner expression symbolically given the children

            # If Terminal, evaluate straight
            if self.is_terminal():
                return self._value.eval_str()

            # Else, evaluate all children first
            return self._value.eval_str(*(child.eval_str() for child in self._children))

        def add_child(self, child_value: 'ParseTree.Node') -> None:
            """
            Add a node to the node's list of children.

            :param child_value: The node to add.
            :raises NodeTerminationException if one attempts to add a child to a terminal node.
            """

            if self.is_terminal():
                raise NodeTerminationException

            self._children.append(child_value)

        def arity(self) -> int:
            """
            Defines the number of child nodes a node has. Equivalent to the number of arguments taken by the enclosed operation.

            :return: An integer arity of the node's inner value/operation.
            """

            return self._value.arity()

        def is_terminal(self) -> bool:
            """
            Indicates whether the node is a terminal node.

            :return: A boolean indicating whether the arity of the node is 0.
            """

            return self.arity() == 0

        def get_depth(self) -> int:
            """
            Gets the depth of the subtree rooted at this node.

            :return: An int representing the depth of the subtree rooted at this node.
            """

            # If a terminal: has only singular depth
            if self.is_terminal():
                return 1

            # Else: depth is 1 + <depth of deepest child>
            return 1 + max([k.get_depth() for k in self._children])

        def __str__(self, level: int = 0, max_depth: int = 0) -> str:
            """
            String representation of the subtree rooted at this node, based on its level in a hierarchy.

            :param level: (Optional) Integer level of the node in a hierarchy.
            :return: A string representation of the subtree rooted at this node, based on its level in a hierarchy.
            """

            prefix: str = ""
            if max_depth > 0:
                rep_chars: int = 1 + floor(log(max_depth, 10))
                rep_buffer: int = rep_chars - len(str(level))
                prefix = str(level) + " " * rep_buffer + ": "

            string_builder = prefix + "\t" * level + self._value.__str__() + "\n"
            for child in self._children:
                string_builder += child.__str__(level + 1, max_depth=max_depth)
            return string_builder

    @classmethod
    def random(cls, max_depth: int, terminal_set: List[Terminal], operator_set: List[Operator]) -> 'ParseTree':
        """
        Method which produces a random non-trivial Parse Tree based on certain criteria.

        :param max_depth: The integer maximum depth of the tree
        :param terminal_set: The terminal set used to construct the tree.
        :param operator_set: The operator set used to construct the tree.
        :return: A parse tree built from pseudorandom combinations of terminals and operators of bounded depth.
        :raises InvalidDepthException: if max_depth is less than 2.
        """

        # A tree cannot be trivial
        if max_depth < 2:
            raise InvalidDepthException

        # Generate a non-terminal root from one atom in the operator set
        root_atom: Operator = random.choice(operator_set).instance()
        root: ParseTree.Node = ParseTree.Node(root_atom)

        # fill the children according to the remaining levels
        cls._fill_level(root, max_depth - 1, terminal_set, operator_set)

        return ParseTree(cls.__create_key, root)

    @classmethod
    def _fill_level(cls, root: Node, rem_levels: int, terminal_set: List[Terminal],
                    operator_set: List[Operator]) -> None:
        """
        Helper method to fill children randomly given a specific number of levels and terminal and operator sets.

        :param root: The root node whose children must be filled.
        :param rem_levels: The number of remaining levels to fill.
        :param terminal_set: The terminal set used to construct the child nodes.
        :param operator_set: The operator set used to construct the child nodes.
        """

        if rem_levels == 0:
            return

        # the number of children for the node
        n_child: int = root.arity()

        # the selection set is both terminals and operators, unless there is only 1 remaining level, in which case it
        # is only terminals
        child_atom_options: List[Atom] = terminal_set if rem_levels < 2 else terminal_set + operator_set

        # for each argument/child
        for i in range(n_child):
            child_atom = random.choice(child_atom_options).instance()
            new_child: ParseTree.Node = ParseTree.Node(child_atom)
            root.add_child(new_child)
            cls._fill_level(new_child, rem_levels - 1, terminal_set, operator_set)

    def __init__(self, key: object, root: Node) -> None:
        """
        Private constructor for a Parse Tree. For internal use only. Use random() to create a new Tree for use.

        :param key: The key used to establish class equality.
        :param root: The root of the Tree
        """

        if not (key == ParseTree.__create_key):
            raise InvalidParseTreeGenerationException

        self._root = root

    def eval(self, symbolic: bool = False, **kwargs):
        if symbolic:
            return self._root.eval_str()
        return self._root.eval(**kwargs)

    def get_depth(self):
        """
        Get the depth of Parse Tree.

        :return: An integer depth of the subtree rooted at the Parse Tree's root.
        """
        return self._root.get_depth()

    def __str__(self):
        """
        A string representation of the Parse Tree.

        :return: A string representation of the subtree rooted at the Parse Tree's root.
        """
        return self._root.__str__(max_depth=self._root.get_depth())


# EXCEPTIONS

class InvalidDepthException(Exception):
    def __init__(self):
        super().__init__("The maximum tree depth must be an integer strictly greater than 1.")


class NodeTerminationException(Exception):
    def __init__(self):
        super().__init__("A child node cannot be added to a terminal node.")


class InvalidParseTreeGenerationException(Exception):
    def __init__(self):
        super().__init__("Make use of the random() method to create a new, random tree.")

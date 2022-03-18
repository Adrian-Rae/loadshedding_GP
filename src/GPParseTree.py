from typing import List, cast, Tuple

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

        def copy(self) -> 'ParseTree.Node':
            new_node = ParseTree.Node(self._value.copy())
            for child in self._children:
                new_node.add_child(child.copy())
            return new_node

        def eval(self, symbolic=False, **kwargs):
            if symbolic:
                return self._eval_str()
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

        def _eval_str(self) -> str:
            # Evaluate the inner expression symbolically given the children

            # If Terminal, evaluate straight
            if self.is_terminal():
                return self._value.eval_str()

            # Else, evaluate all children first
            return self._value.eval_str(*(child._eval_str() for child in self._children))

        def add_child(self, child_value: 'ParseTree.Node') -> None:
            """
            Add a node to the node's list of children.

            :param child_value: The node to add.
            :raises NodeTerminationException if one attempts to add a child to a terminal node.
            """

            if self.is_terminal():
                raise NodeTerminationException

            self._children.append(child_value)

        def set_child(self, child_node: 'ParseTree.Node', index: int) -> None:
            self._children[index] = child_node

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

        def _linearize(self, exclude_first=False) -> List['ParseTree.Node']:
            buffer = [self] if not exclude_first else []
            for child in self._children:
                buffer += child._linearize()
            return buffer

        def random_node(self, exclude_first=False) -> 'ParseTree.Node':
            return random.choice(self._linearize(exclude_first=exclude_first))

        def get_node_lineage(self, target: 'ParseTree.Node', parent: 'ParseTree.Node' = None):
            # get a node, along with it's parent and child index
            if self == target:
                index = None
                if parent is not None:
                    index = parent._children.index(self)
                return self, parent, index

            for child in self._children:
                c, p, i = child.get_node_lineage(target, parent=self)
                if c is not None:
                    return c, p, i

            return None, None, None

        def depth_of(self, node: 'ParseTree.Node', level: int = 1):
            if self == node:
                return level
            for child in self._children:
                cd = child.depth_of(node, level + 1)
                if cd is not None:
                    return cd
            return None

        def __or__(self, other):
            # define self | other to mean independence => not descendants
            return not ((self < other) or (other < self))

        def __lt__(self, other: 'ParseTree.Node'):
            # define self < other iff self is a descendant of other => applies transitively
            if self == other:
                return False
            base = False
            for child in other._children:
                if self == child:
                    return True
                else:
                    base = base or (self < child)
            return base

        def __gt__(self, other: 'ParseTree.Node'):
            # define self > other iff other < self
            return other < self

        def __rshift__(self, other: 'ParseTree.Node'):
            # define self >> other if self is added as a child to other
            other.add_child(self)

        def __lshift__(self, other: 'ParseTree.Node'):
            # define self << other if other is added as a child to self
            self.add_child(other)

        def __str__(self) -> str:
            return self.eval(symbolic=True)

    @classmethod
    def random(cls, max_depth: int, terminal_set: List[Terminal], operator_set: List[Operator], force_trivial: bool = False) -> 'ParseTree':
        """
        Method which produces a random non-trivial Parse Tree based on certain criteria.

        :param force_trivial: Flag to allow the generation of a trivial tree
        :param max_depth: The integer maximum depth of the tree
        :param terminal_set: The terminal set used to construct the tree.
        :param operator_set: The operator set used to construct the tree.
        :return: A parse tree built from pseudorandom combinations of terminals and operators of bounded depth.
        :raises InvalidDepthException: if max_depth is less than 2.
        """

        # A tree cannot be trivial if not specified
        if max_depth < 2 and not force_trivial:
            raise InvalidDepthException

        choice_set_root = operator_set if (max_depth > 1) else terminal_set

        # Generate a non-terminal root from one atom in the operator set
        root_atom: Operator = random.choice(choice_set_root).instance()
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
            root << new_child
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

    def get_root(self):
        return self._root

    def eval(self, symbolic: bool = False, **kwargs):
        return self._root.eval(symbolic=symbolic, **kwargs)

    def get_depth(self):
        """
        Get the depth of Parse Tree.

        :return: An integer depth of the subtree rooted at the Parse Tree's root.
        """
        return self._root.get_depth()

    def random_node(self, exclude_root=False):
        return self._root.random_node(exclude_first=exclude_root)

    def _random_node_pair(self) -> Tuple[Node, Node]:
        # returns a pair of non-descendant nodes
        # choose one node that isn't the root, as all nodes descend from root
        n1 = self.random_node(exclude_root=True)

        # choose a second node provided they aren't dependent
        timeout = 10
        counter = 0
        while counter < timeout:
            n2 = self.random_node(exclude_root=True)
            if n1 | n2:
                return n1, n2
            counter += 1

        # just return root combo as they are by definition, non-descendant
        return self._root, self._root

    def _get_node_lineage(self, target: Node, parent: Node = None):
        return self._root.get_node_lineage(target, parent=parent)

    def replace_node(self, root: Node, replacement: Node):

        # find lineage of the root
        n, p, i = self._get_node_lineage(root)

        if p is None:
            # root is root of whole tree
            self._root = replacement
            return

        # else adjust accordingly
        p.set_child(replacement, i)

    def _swap_nodes(self, n1: Node, n2: Node):
        # can only swap two non-descendant nodes

        # if identical, no need to do anything
        if n1 == n2:
            return

        # find lineages of both
        n1, p1, i1 = self._get_node_lineage(n1)
        n2, p2, i2 = self._get_node_lineage(n2)

        # disconnect existing and set to new
        for i, p, new in [(i1, p1, n2), (i2, p2, n1)]:
            p.set_child(new, i)

    def swap_random_node_pair(self):
        self._swap_nodes(*self._random_node_pair())

    def copy(self) -> 'ParseTree':
        return ParseTree(self.__create_key, self._root.copy())

    def depth_of(self, node: Node):
        return self._root.depth_of(node)

    def __str__(self):
        """
        A string representation of the Parse Tree.

        :return: A string representation of the subtree rooted at the Parse Tree's root.
        """
        return self._root.__str__()


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

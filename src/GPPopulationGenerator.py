import threading
from enum import Enum
from math import floor

from GPParseTree import *


class PopulationGenerator:
    """
        Class that handles generation of an initial population of GPParseTree objects based on selected criteria.
    """

    class Method(Enum):
        """
            Nested class that specifies a growth method scheme for generating a population.
        """

        GROW = 0
        FULL = 1
        RAMPED = 2

    def __init__(self, terminal_set: List[Terminal], operator_set: List[Operator], parallelization: int = 1) -> None:
        """
            Constructor for the generator.

            :param terminal_set: The terminal set used in an individual's construction.
            :param operator_set: The operator set used in an individual's construction.
            :raises InvalidTerminalSetSizeException: if the terminal set given is empty.
            :raises InvalidOperatorSetSizeException: if the operator set given is empty.
        """

        # Safety checks on initial criteria
        if len(terminal_set) == 0:
            raise InvalidTerminalSetSizeException
        if len(operator_set) == 0:
            raise InvalidOperatorSetSizeException

        self._terminal_set = terminal_set
        self._operator_set = operator_set
        self._parallelization = parallelization
        self._population_buffer = []
        self._serial_lock = threading.Lock()
        self._generating_lock = threading.Lock()

    def generate(self, size: int, max_depth: int, method: Method = Method.GROW, force_trivial: bool = False,
                 force_fairness: bool = False) -> List[ParseTree]:

        with self._generating_lock:
            population_split = [sum(p) for p in np.array_split([1 for _ in range(size)], self._parallelization)]
            threads = []
            for sub_pop_range in population_split:
                if sub_pop_range == 0:
                    continue
                args = (sub_pop_range, max_depth, method, force_trivial, force_fairness)
                active_thread = threading.Thread(
                    target=self._generate_serial,
                    args=args,
                    daemon=True
                )
                threads.append(active_thread)

            # wait for responses
            for thread in threads:
                thread.start()
                thread.join()

            # get the buffer value
            new_pop = self._population_buffer

            #clear the buffer
            self._population_buffer = []

            return new_pop

    def _generate_serial(self, size: int, max_depth: int, method: Method = Method.GROW, force_trivial: bool = False,
                         force_fairness: bool = False):
        """
        Method to generate a population of ParseTree individuals based on specified criteria.

        :param force_trivial: flag to allow the generation of trivial trees
        :param size: The size of the initial population to be generated.
        :param max_depth: The maximum depth of an individual. For FULL and RAMPED growth methods, this is the target depth.
        :param method: (Optional) The desired generation method: GROW, FULL or RAMPED. Defaults to GROW.
        :return: A list of ParseTree individuals.
        :raises InvalidPopulationGenerationMethodException: if an invalid generation method is given.
        :raises InvalidPopulationSizeException: if the population size specified is non-positive.
        :raises InvalidDepthException: if max_depth is less than 2.
        """

        # Safety checks on initial criteria
        if method is None:
            raise InvalidPopulationGenerationMethodException
        if size < 1:
            raise InvalidPopulationSizeException
        if max_depth < 2 and not force_trivial:
            raise InvalidDepthException

        # GROW METHOD
        if method == PopulationGenerator.Method.GROW:

            population: List[ParseTree] = []

            # Simply return n randomly generated trees with the specified maximum depth
            for i in range(size):
                new_member: ParseTree = ParseTree.random(
                    max_depth,
                    self._terminal_set,
                    self._operator_set,
                    force_trivial=force_trivial,
                    fairness=force_fairness
                )
                population.append(new_member)

            with self._serial_lock:
                self._population_buffer += population
                return

        # FULL METHOD
        elif method == PopulationGenerator.Method.FULL:

            # Generate a population of n members with identical (maximum) depth
            population: List[ParseTree] = []

            while True:

                # Create a new member
                new_member: ParseTree = ParseTree.random(
                    max_depth,
                    self._terminal_set,
                    self._operator_set,
                    force_trivial=force_trivial,
                    fairness=force_fairness
                )

                # If the depth is desired
                if new_member.get_depth() == max_depth:
                    # keep the member
                    population.append(new_member)

                # If the population size is reached, terminate
                if len(population) == size:
                    break

            # Return the population
            with self._serial_lock:
                self._population_buffer += population
                return

        # RAMPED HALF-AND-HALF
        elif method == PopulationGenerator.Method.RAMPED:

            # Strata of depth
            depth_strata = range(2, 1 + max_depth)

            # Get the division factor : each depth between 2 and the specified maximum is allocated an equal number
            # of individuals, with overflows being allocated randomly.
            n_divisions: int = (max_depth - 1)
            n_local: int = floor(size / n_divisions)

            # number to be generated with each method
            n_half: int = floor(n_local / 2)

            # residual, level-wise members
            n_res_level: int = n_local - n_half

            # residue of main population : those not assigned to a strata
            n_res: int = size - n_local * n_divisions

            # Generate a population of n members fair depth distribution
            population: List[ParseTree] = []

            # if subpopulations exists
            if n_local > 0:
                # generate half the local with full for each depth strata
                for depth in depth_strata:
                    if n_half > 0:
                        # generate by full method
                        population += self.generate(
                            n_half,
                            depth,
                            method=PopulationGenerator.Method.FULL,
                            force_trivial=force_trivial
                        )
                        # generate by grow method
                        population += self.generate(
                            n_half,
                            depth,
                            method=PopulationGenerator.Method.GROW,
                            force_trivial=force_trivial
                        )
                    if n_res_level > 0:
                        # allocate remaining members on level randomly
                        rem_method = random.choice([PopulationGenerator.Method.GROW, PopulationGenerator.Method.FULL])
                        # generate by grow method
                        population += self.generate(
                            n_res_level,
                            depth,
                            method=rem_method,
                            force_trivial=force_trivial
                        )

            # get a subset of depth strata for the residue members
            residue_strata = random.sample(depth_strata, n_res)

            # repeat the process only for the residue strata - create only individuals each time
            for depth in residue_strata:
                population += self.generate(
                    1,
                    depth,
                    method=PopulationGenerator.Method.FULL,
                    force_trivial=force_trivial
                )

            with self._serial_lock:
                self._population_buffer += population
                return

        # Invalid generation method bound
        raise InvalidPopulationGenerationMethodException


# EXCEPTIONS

class InvalidPopulationSizeException(Exception):
    def __init__(self):
        super().__init__("The population size must be an integer strictly greater than 0.")


class InvalidPopulationGenerationMethodException(Exception):
    def __init__(self):
        super().__init__("The specified population generation method is invalid.")


class InvalidTerminalSetSizeException(Exception):
    def __init__(self):
        super().__init__("A terminal set must contain at least one element.")


class InvalidOperatorSetSizeException(Exception):
    def __init__(self):
        super().__init__("An operator set must contain at least one element.")

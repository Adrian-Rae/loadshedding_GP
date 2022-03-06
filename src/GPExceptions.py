class InvalidDepthException(Exception):
    def __init__(self):
        super().__init__("The maximum tree depth must be an integer strictly greater than 1.")


class NodeTerminationException(Exception):
    def __init__(self):
        super().__init__("A child node cannot be added to a terminal node.")


class NonTerminationBindingException(Exception):
    def __init__(self):
        super().__init__("A non-terminal node must have at least one child node bound.")


class InvalidPopulationSizeException(Exception):
    def __init__(self):
        super().__init__("The population size must be an integer strictly greater than 0.")


class InvalidPopulationGenerationMethodException(Exception):
    def __init__(self):
        super().__init__("The specified population generation method is invalid.")

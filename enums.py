from enum import Enum


class Results(Enum):
    """
    keys of values returned by the move function of the simulator. use them in order to gather information
    on the current state of the simulation.
    """
    COLLISION = 1
    PERCENTAGE_IN_TARGET = 2
    FRAME = 3
    SIMULATION_TIMEOUT = 4
    IN_BOUNDS = 5

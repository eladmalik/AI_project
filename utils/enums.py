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


class SensorDirection(Enum):
    LEFT = 0
    FRONT = 1
    RIGHT = 2
    BACK = 3
    FRONTLEFT = 4
    FRONTRIGHT = 5
    BACKLEFT = 6
    BACKRIGHT = 7


class Movement(Enum):
    """
    Indicates the forward/backward movement of a car
    """
    FORWARD = 1
    NEUTRAL = 0
    BACKWARD = -1
    BRAKE = 2


class Steering(Enum):
    """
    Indicates the side which the steering wheel is rotated to
    """
    LEFT = 1
    NEUTRAL = 0
    RIGHT = -1
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
    SUCCESS = 6
    DISTANCE_TO_TARGET = 7
    ANGLE_TO_TARGET = 8


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


class StatsType(Enum):
    LAST_REWARD = 0
    TOTAL_REWARD = 1
    DISTANCE_TO_TARGET = 2
    PERCENTAGE_IN_TARGET = 3
    ANGLE_TO_TARGET = 4
    SUCCESS = 5
    COLLISION = 6
    IS_DONE = 7
    I_EPISODE = 8
    I_STEP = 9
    GENERATION = 10

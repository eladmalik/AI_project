from enum import Enum
from sim.car import Movement, Steering

class CarActions(Enum):
    (Movement.NEUTRAL, Steering.NEUTRAL) = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (Movement.NEUTRAL, Steering.LEFT) = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (Movement.NEUTRAL, Steering.RIGHT) = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (Movement.FORWARD, Steering.NEUTRAL) = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    (Movement.FORWARD, Steering.LEFT) = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    (Movement.FORWARD, Steering.RIGHT) = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    (Movement.BACKWARD, Steering.NEUTRAL) = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    (Movement.BACKWARD, Steering.LEFT) = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    (Movement.BACKWARD, Steering.RIGHT) = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    (Movement.BRAKE, Steering.NEUTRAL) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    (Movement.BRAKE, Steering.LEFT) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    (Movement.BRAKE, Steering.RIGHT) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

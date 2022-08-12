from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict
import math

from parking_lot import ParkingLot


class Results(Enum):
    """
    keys of values returned by the move function of the simulator. use them in order to gather information
    on the current state of the simulation.
    """
    COLLISION = 1
    PERCENTAGE_IN_TARGET = 2
    UNOCCUPIED_PERCENTAGE = 3


class RewardAnalyzer(ABC):
    @abstractmethod
    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]):
        """
        This function gets a current parking lot state and the results which this parking lot has,
        and returns a reward value
        """
        ...


class Analyzer1(RewardAnalyzer):
    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    MAX_IN_TARGET_REWARD = 100  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 1  # higher => less reward as the car is faster while in the target

    def __init__(self):
        self.max_distance_reward = Analyzer1.MAX_DISTANCE_TO_TARGET_REWARD
        self.max_in_target_reward = Analyzer1.MAX_IN_TARGET_REWARD
        self.max_angle_to_target_reward = Analyzer1.MAX_ANGLE_TO_TARGET_REWARD
        self.velocity_penalty_factor = Analyzer1.VELOCITY_PENALTY_FACTOR

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]):
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        # as the car in getting closer to the target, the reward increases
        distance_reward = self.max_distance_reward / (distance_to_target + 1)

        # as the car in inside the target, the reward increases
        in_target_reward = self.max_in_target_reward * results[Results.PERCENTAGE_IN_TARGET]

        # penalty for speeding in the target cell
        in_target_reward = in_target_reward / (1 / (
                self.VELOCITY_PENALTY_FACTOR * parking_lot.car_agent.velocity.magnitude()) + 1)

        angle_to_target_reward = 0
        if results[Results.PERCENTAGE_IN_TARGET] > 0:
            angle_to_target_reward = self.max_angle_to_target_reward * abs(math.cos(abs(math.radians(
                parking_lot.car_agent.rotation) - math.radians(parking_lot.target_park.rotation))))

        return distance_reward + in_target_reward + angle_to_target_reward

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
    ID: int

    @abstractmethod
    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> float:
        """
        This function gets a current parking lot state and the results which this parking lot has,
        and returns a reward value
        """
        ...


class Analyzer1(RewardAnalyzer):
    ID = 0
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.1  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 100  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 0.3  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> float:
        if results[Results.COLLISION]:
            return self.COLLISION_PENALTY
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        # as the car in getting closer to the target, the reward increases
        distance_reward = self.MAX_DISTANCE_TO_TARGET_REWARD / (
                (self.DISTANCE_REWARD_FACTOR * distance_to_target) + 1)

        # as the car in inside the target, the reward increases
        in_target_reward = self.MAX_IN_TARGET_REWARD * results[Results.PERCENTAGE_IN_TARGET]

        # penalty for speeding in the target cell
        in_target_reward = in_target_reward * (1 / ((
                                                            self.VELOCITY_PENALTY_FACTOR * parking_lot.car_agent.velocity.magnitude()) + 1))

        angle_to_target_reward = 0
        if results[Results.PERCENTAGE_IN_TARGET] > 0:
            angle_to_target_reward = self.MAX_ANGLE_TO_TARGET_REWARD * abs(math.cos(abs(math.radians(
                parking_lot.car_agent.rotation) - math.radians(parking_lot.target_park.rotation))))

        return distance_reward + in_target_reward + angle_to_target_reward

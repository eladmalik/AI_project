from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple
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
    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        """
        This function gets a current parking lot state and the results which this parking lot has,
        and returns a reward value
        """
        ...


class Analyzer(RewardAnalyzer):
    ID = 0
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 1  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 100  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 0.6  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        if results[Results.COLLISION]:
            return self.COLLISION_PENALTY, True
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

        return distance_reward + in_target_reward + angle_to_target_reward, False


class AnalyzerPenaltyOnStanding(RewardAnalyzer):
    ID = 1
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 1  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 100  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 0.6  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent

    STANDING_STILL_PENALTY = -200
    ZERO_VELOCITY_EPSILON = 0

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        if results[Results.COLLISION]:
            return self.COLLISION_PENALTY, True
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        if results[
            Results.PERCENTAGE_IN_TARGET] <= 0 and parking_lot.car_agent.velocity.magnitude() <= \
                self.ZERO_VELOCITY_EPSILON:
            return self.STANDING_STILL_PENALTY, False

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

        return distance_reward + in_target_reward + angle_to_target_reward, False


class AnalyzerStopOnTarget(RewardAnalyzer):
    ID = 2
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.05  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 1000  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 2  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent

    STANDING_STILL_PENALTY = -200
    ZERO_VELOCITY_EPSILON = 0.05

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        if results[Results.COLLISION]:
            return self.COLLISION_PENALTY, True
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        # if results[
        #     Results.PERCENTAGE_IN_TARGET] <= 0 and parking_lot.car_agent.velocity.magnitude() <= \
        #         self.ZERO_VELOCITY_EPSILON:
        #     return self.STANDING_STILL_PENALTY, False

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

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() == 0:
            return distance_reward + in_target_reward + angle_to_target_reward, True

        return distance_reward + in_target_reward + angle_to_target_reward, False


class AnalyzerDistanceCritical(RewardAnalyzer):
    ID = 3
    # keep every argument greater than 0

    MIN_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.001  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 1000  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 2  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent

    STANDING_STILL_PENALTY = -200
    ZERO_VELOCITY_EPSILON = 0.05

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        if results[
            Results.PERCENTAGE_IN_TARGET] <= 0 and parking_lot.car_agent.velocity.magnitude() <= \
                self.ZERO_VELOCITY_EPSILON:
            return self.STANDING_STILL_PENALTY, False

        # as the car in getting closer to the target, the reward increases
        if results[Results.PERCENTAGE_IN_TARGET] > 0:
            distance_reward = 0
        else:
            distance_reward = self.MIN_DISTANCE_TO_TARGET_REWARD / (
                    (
                            self.DISTANCE_REWARD_FACTOR * distance_to_target) + 1) - self.MIN_DISTANCE_TO_TARGET_REWARD

        # as the car in inside the target, the reward increases
        in_target_reward = self.MAX_IN_TARGET_REWARD * results[Results.PERCENTAGE_IN_TARGET]

        # penalty for speeding in the target cell
        in_target_reward = in_target_reward * (1 / ((
                                                            self.VELOCITY_PENALTY_FACTOR * parking_lot.car_agent.velocity.magnitude()) + 1))

        angle_to_target_reward = 0
        if results[Results.PERCENTAGE_IN_TARGET] > 0:
            angle_to_target_reward = self.MAX_ANGLE_TO_TARGET_REWARD * abs(math.cos(abs(math.radians(
                parking_lot.car_agent.rotation) - math.radians(parking_lot.target_park.rotation))))

        if results[Results.COLLISION]:
            return min(0, distance_reward + in_target_reward + angle_to_target_reward), True

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() == 0:
            return distance_reward + in_target_reward + angle_to_target_reward, True

        return distance_reward + in_target_reward + angle_to_target_reward, False


class AnalyzerCollisionReduceNearTarget(RewardAnalyzer):
    ID = 4
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.05  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 1000  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 2  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent
    COLLISION_PENALTY_FACTOR = 0.003  # lower => less penalty for far collisions from target

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if results[Results.COLLISION]:
            return math.erf(self.COLLISION_PENALTY_FACTOR * distance_to_target) * self.COLLISION_PENALTY, True

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

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() == 0:
            return distance_reward + in_target_reward + angle_to_target_reward, True

        return distance_reward + in_target_reward + angle_to_target_reward, False


class AnalyzerNoCollision(RewardAnalyzer):
    ID = 5
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.05  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 1000  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 2  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent
    COLLISION_PENALTY_FACTOR = 0.003  # lower => less penalty for far collisions from target

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
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

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() == 0:
            return distance_reward + in_target_reward + angle_to_target_reward, True

        return distance_reward + in_target_reward + angle_to_target_reward, results[Results.COLLISION]


class AnalyzerNoCollisionNoDistanceReward(RewardAnalyzer):
    ID = 6
    # keep every argument greater than 0

    MAX_DISTANCE_TO_TARGET_REWARD = 100  # higher => more reward as the car is closer to the target
    DISTANCE_REWARD_FACTOR = 0.05  # lower => more reward for far distances

    MAX_IN_TARGET_REWARD = 1000  # higher => more reward as the car is more inside target
    MAX_ANGLE_TO_TARGET_REWARD = 50  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 2  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -100  # lower => more penalty for the agent
    COLLISION_PENALTY_FACTOR = 0.003  # lower => less penalty for far collisions from target

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)

        # as the car in inside the target, the reward increases
        in_target_reward = self.MAX_IN_TARGET_REWARD * results[Results.PERCENTAGE_IN_TARGET]

        # penalty for speeding in the target cell
        in_target_reward = in_target_reward * (1 / ((
                                                            self.VELOCITY_PENALTY_FACTOR * parking_lot.car_agent.velocity.magnitude()) + 1))

        angle_to_target_reward = 0
        if results[Results.PERCENTAGE_IN_TARGET] > 0:
            angle_to_target_reward = self.MAX_ANGLE_TO_TARGET_REWARD * abs(math.cos(abs(math.radians(
                parking_lot.car_agent.rotation) - math.radians(parking_lot.target_park.rotation))))

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() == 0:
            return in_target_reward + angle_to_target_reward, True

        return in_target_reward + angle_to_target_reward, False


class AnalyzerAccumulating(RewardAnalyzer):
    ID = 7
    COLLISION_PENALTY = -500
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 600
    PARKED_REWARD = 1000

    def __init__(self):
        # self.distances = [2 ** i for i in range(9)] + [x for x in range(300, 900, 100)]
        self.distances = [1, 33, 66, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900]
        self.rewards = [(x // 100) + 1 for x in self.distances]
        self.distances.reverse()
        self.in_parking = False
        self.best_distance_index = 0

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if results[Results.COLLISION]:
            return math.erf(self.COLLISION_PENALTY_FACTOR * current_distance) * self.COLLISION_PENALTY, True
        reward = 0
        while self.best_distance_index < len(self.distances) and \
                self.distances[self.best_distance_index] >= current_distance:
            reward += self.rewards[self.best_distance_index]
            self.rewards[self.best_distance_index] = 0
            self.best_distance_index += 1
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward += self.PARKED_REWARD
            return reward, True
        return reward, False

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
    FRAME = 3
    SIMULATION_TIMEOUT = 4


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
    MAX_ANGLE_TO_TARGET_REWARD = 200  # higher => more reward as the car more aligned with the target
    VELOCITY_PENALTY_FACTOR = 0.08  # higher => less reward as the car is faster while in the target

    COLLISION_PENALTY = -30  # lower => more penalty for the agent
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


class AnalyzerAccumulating2(RewardAnalyzer):
    ID = 8
    COLLISION_PENALTY = -500
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 600
    PARKED_REWARD = 1000

    def __init__(self):
        # self.distances = [2 ** i for i in range(9)] + [x for x in range(300, 900, 100)]
        self.best_distance = -float("inf")
        self.in_parking = False

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if results[Results.COLLISION]:
            return math.erf(self.COLLISION_PENALTY_FACTOR * current_distance) * self.COLLISION_PENALTY, True
        reward = 0
        if current_distance < self.best_distance:
            reward += 3
            self.best_distance = current_distance
        else:
            reward -= 1
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward += self.PARKED_REWARD
            return reward, True
        return reward, False


class AnalyzerAccumulating3(RewardAnalyzer):
    ID = 8
    COLLISION_PENALTY = -10
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 600
    PARKED_REWARD = 1000

    def __init__(self):
        # self.distances = [2 ** i for i in range(9)] + [x for x in range(300, 900, 100)]
        self.distances = [1, 33, 66, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900]
        self.rewards = [(x // 100) + 1 for x in self.distances[:-1]]
        self.distances.reverse()
        self.in_parking = False
        self.init = False
        self.last_distance = float("inf")
        self.outside_circle = 0

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if not self.init:
            self.init = True
            self.last_distance = current_distance
            while current_distance < self.distances[self.outside_circle]:
                self.outside_circle += 1
        if results[Results.COLLISION]:
            return math.erf(self.COLLISION_PENALTY_FACTOR * current_distance) * self.COLLISION_PENALTY, True
        reward = 0

        if current_distance < self.last_distance:
            while self.outside_circle < len(self.distances) and \
                    current_distance < self.distances[self.outside_circle]:
                reward += self.rewards[self.outside_circle]
                self.outside_circle += 1
        elif current_distance > self.last_distance:
            while self.outside_circle > 0 and current_distance > self.distances[self.outside_circle - 1]:
                reward -= self.rewards[self.outside_circle - 1]
                self.outside_circle -= 1
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward += self.PARKED_REWARD
            return reward, True
        self.last_distance = current_distance
        return reward, False


class AnalyzerAccumulating4(RewardAnalyzer):
    ID = 9
    COLLISION_PENALTY = -100
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 1000
    PARKED_REWARD = 2000

    def __init__(self):
        self.distances = [d for d in range(1200, 0, -2)]
        self.rewards = [2 for _ in range(len(self.distances))]
        self.in_parking = False
        self.init = False
        self.last_distance = float("inf")
        self.outside_circle = 0

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if not self.init:
            self.init = True
            self.last_distance = current_distance
            while current_distance < self.distances[self.outside_circle]:
                self.outside_circle += 1
        if results[Results.COLLISION]:
            return math.erf(self.COLLISION_PENALTY_FACTOR * current_distance) * self.COLLISION_PENALTY, True
        reward = 0

        if current_distance < self.last_distance:
            while self.outside_circle < len(self.distances) and \
                    current_distance < self.distances[self.outside_circle]:
                reward += self.rewards[self.outside_circle]
                self.outside_circle += 1
        elif current_distance > self.last_distance:
            while self.outside_circle > 0 and current_distance > self.distances[self.outside_circle - 1]:
                reward -= self.rewards[self.outside_circle - 1]
                self.outside_circle -= 1
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward += self.PARKED_REWARD
            return reward, True
        self.last_distance = current_distance
        return reward, False


class AnalyzerAccumulating5(RewardAnalyzer):
    ID = 10
    COLLISION_PENALTY = -100
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 1000
    PARKED_REWARD = 2000

    def __init__(self):
        self.distances = [d for d in range(1200, 0, -2)]
        self.rewards = [2] * len(self.distances)
        self.in_parking = False
        self.init = False
        self.last_distance = float("inf")
        self.outside_circle = 0

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        if not self.init:
            self.init = True
            self.last_distance = current_distance
            while current_distance < self.distances[self.outside_circle]:
                self.outside_circle += 1
        if results[Results.COLLISION]:
            return (math.erf(self.COLLISION_PENALTY_FACTOR * current_distance) *
                    self.COLLISION_PENALTY) / 1200, \
                   True

        if results[Results.PERCENTAGE_IN_TARGET] < 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            return -0.001 * (current_distance / 1200), False
        reward = 0

        if current_distance < self.last_distance:
            while self.outside_circle < len(self.distances) and \
                    current_distance < self.distances[self.outside_circle]:
                parking_vec = parking_lot.target_park.left - parking_lot.target_park.location
                car_vec = parking_lot.target_park.location - parking_lot.car_agent.front
                angle = car_vec.angle_to(parking_vec)

                reward += self.rewards[self.outside_circle] * abs(math.cos(math.radians(angle)))
                self.outside_circle += 1
        elif current_distance > self.last_distance:
            while self.outside_circle > 0 and current_distance > self.distances[self.outside_circle - 1]:
                reward -= self.rewards[self.outside_circle - 1]
                self.outside_circle -= 1
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward += self.PARKED_REWARD
            return reward / 1200, True
        self.last_distance = current_distance
        return reward / 1200, False


class AnalyzerAccumulating6(RewardAnalyzer):
    ID = 11
    COLLISION_PENALTY = -200
    COLLISION_PENALTY_FACTOR = 0.003

    IN_PARKING_REWARD = 1000
    PARKED_REWARD = 0.1
    ANGLE_IN_PARKING_REWARD = 20
    DISTANCE_IN_PARKING_REWARD = 20

    def __init__(self):
        self.in_parking = False
        self.init = False
        self.max_distance_inside_target = 1200
        self.last_distance = float("inf")
        self.outside_circle = 0
        self.distances = [d for d in range(1200, 200, -2)]
        self.rewards = [2] * len(self.distances)

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        current_distance = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        reward = 0
        done = False
        if not self.init:
            self.init = True
            self.max_distance_inside_target = math.sqrt(
                (parking_lot.target_park.width ** 2) + (parking_lot.target_park.height ** 2))
            self.last_distance = current_distance
            while self.outside_circle < len(self.distances) and current_distance < self.distances[ \
                    self.outside_circle]:
                self.outside_circle += 1

        if current_distance < self.last_distance:
            while self.outside_circle < len(self.distances) and \
                    current_distance < self.distances[self.outside_circle]:
                reward += self.rewards[self.outside_circle]
                self.outside_circle += 1
        elif current_distance > self.last_distance:
            while self.outside_circle > 0 and current_distance > self.distances[self.outside_circle - 1]:
                reward -= self.rewards[self.outside_circle - 1]
                self.outside_circle -= 1

        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and not self.in_parking:
            self.in_parking = True
            reward += self.IN_PARKING_REWARD
        if results[Results.PERCENTAGE_IN_TARGET] >= 1 and parking_lot.car_agent.velocity.magnitude() <= 0:

            parking_angle_diff = abs(
                math.radians(parking_lot.target_park.rotation) -
                math.radians(parking_lot.car_agent.rotation))
            if abs(math.cos(parking_angle_diff)) > 0.866:  # 30 degrees
                reward += self.PARKED_REWARD
                reward += self.ANGLE_IN_PARKING_REWARD * abs(math.cos(parking_angle_diff))
                reward += self.DISTANCE_IN_PARKING_REWARD * (
                        1 - (current_distance / self.max_distance_inside_target))

        if results[Results.PERCENTAGE_IN_TARGET] <= 0 and parking_lot.car_agent.velocity.magnitude() <= 0:
            reward -= 0.1 * (current_distance / 1200)

        if results[Results.PERCENTAGE_IN_TARGET] <= 0 and self.in_parking:
            self.in_parking = False
            reward -= self.IN_PARKING_REWARD

        if results[Results.COLLISION]:
            reward += self.COLLISION_PENALTY
            done = True

        return reward / 1200, done


class AnalyzerNew(RewardAnalyzer):
    ID = 12

    STOP_COUNT = 360
    DISTANCE_FACTOR = 2
    MAX_DISTANCE_FACTOR = 2
    DONE_FACTOR = 10
    MIN_DISTANCE_FROM_TARGET = 0.5

    FINAL_DISTANCE_REWARD = 50
    FINAL_DISTANCE_FACTOR = 0.1

    FINAL_IN_TARGET_BONUS = 2

    TIMEOUT_PENALTY = 150

    COLLISION_PENALTY = 1000

    def __init__(self):
        self.stop_history = 0
        self.current_lot = None

    def __distance_from_target(self):
        return self.current_lot.car_agent.location.distance_to(
            self.current_lot.target_park.location) / self.current_lot.car_agent.height

    def __reward_from_distance(self):
        # TODO add search for the shortest path to target, with respect to obstacles
        # max_distance = math.sqrt((self.current_lot.width ** 2) + (self.current_lot.height ** 2)) / \
        #                self.current_lot.car_agent.height
        #
        # max_dist_factored = max_distance ** self.MAX_DISTANCE_FACTOR
        dist_factored = self.__distance_from_target() ** self.DISTANCE_FACTOR

        return -dist_factored

    def __reward_from_angle(self):
        parking_vec = self.current_lot.target_park.back - self.current_lot.target_park.location
        car_vec = self.current_lot.car_agent.front - self.current_lot.car_agent.location
        angle = 2 - abs(math.cos(2 * math.radians(car_vec.angle_to(parking_vec))) - 1)
        return angle

    def analyze(self, parking_lot: ParkingLot, results: Dict[Results, Any]) -> Tuple[float, bool]:
        self.current_lot = parking_lot
        reward = 0
        if parking_lot.car_agent.velocity.magnitude() == 0:
            self.stop_history += 1
        else:
            self.stop_history = 0

        done = self.stop_history > self.STOP_COUNT
        distance = self.__distance_from_target()

        if results[Results.COLLISION]:
            return -self.COLLISION_PENALTY, True

        if not done:
            reward += self.__reward_from_distance()
            if results[Results.SIMULATION_TIMEOUT]:
                reward -= self.TIMEOUT_PENALTY + distance
        else:
            if distance < self.MIN_DISTANCE_FROM_TARGET:
                reward += self.__reward_from_angle()
            reward += self.FINAL_DISTANCE_REWARD / (1 + self.FINAL_DISTANCE_FACTOR * distance)
            if results[Results.PERCENTAGE_IN_TARGET] > 0.5:
                reward *= 2
        return reward, done

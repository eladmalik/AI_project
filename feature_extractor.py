import math
from abc import ABC
from typing import List

from parking_lot import ParkingLot
from proximity_sensor import SensorDirection


class FeatureExtractor(ABC):
    ID: int
    input_num: int

    @staticmethod
    def get_state(self, parking_lot: ParkingLot):
        """
        This function gets the parking lot as an inputs and outputs a list of numerical features which
        indicates the current state of the parking lot
        """
        ...


class Extractor(FeatureExtractor):
    ID = 0
    FEATURES = [
        "Relative X to target",
        "Relative Y to target",
        "Relative Rotation to target (normalized with Cos)",
        "Velocity X",
        "Velocity Y",
        "Acceleration",
        "Steering",
        "Sensor Front",
        "Sensor Back",
        "Sensor Left",
        "Sensor Right"
    ]
    input_num = len(FEATURES)

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        relative_x = parking_lot.target_park.location.x - parking_lot.car_agent.location.x
        relative_y = parking_lot.target_park.location.y - parking_lot.car_agent.location.y
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))
        velocity_x = parking_lot.car_agent.velocity.x
        velocity_y = parking_lot.car_agent.velocity.y
        acceleration = parking_lot.car_agent.acceleration
        steering = float(parking_lot.car_agent.steering)
        sensor_front = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.FRONT])
        sensor_back = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.BACK])
        sensor_left = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.LEFT])
        sensor_right = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.RIGHT])

        return [relative_x, relative_y, relative_rotation, velocity_x, velocity_y, acceleration, steering,
                sensor_front, sensor_back, sensor_left, sensor_right]


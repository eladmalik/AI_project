import math
from abc import ABC
from typing import List

import pygame

from parking_lot import ParkingLot
from proximity_sensor import SensorDirection


class FeatureExtractor(ABC):
    ID: int
    input_num: int

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


class Extractor2(FeatureExtractor):
    ID = 1
    FEATURES = [
        "Relative X to target",
        "Relative Y to target",
        "Distance to target",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
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
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))
        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

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

        return [relative_x, relative_y, distance_to_target, relative_rotation, angle_to_target, velocity_x,
                velocity_y, acceleration, steering, sensor_front, sensor_back, sensor_left, sensor_right]


class Extractor2NoSensors(FeatureExtractor):
    ID = 2
    FEATURES = [
        "Relative X to target",
        "Relative Y to target",
        "Distance to target",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
        "Velocity X",
        "Velocity Y",
        "Acceleration",
        "Steering"
    ]
    input_num = len(FEATURES)

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        relative_x = parking_lot.target_park.location.x - parking_lot.car_agent.location.x
        relative_y = parking_lot.target_park.location.y - parking_lot.car_agent.location.y
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))
        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

        velocity_x = parking_lot.car_agent.velocity.x
        velocity_y = parking_lot.car_agent.velocity.y
        acceleration = parking_lot.car_agent.acceleration
        steering = float(parking_lot.car_agent.steering)

        return [relative_x, relative_y, distance_to_target, relative_rotation, angle_to_target, velocity_x,
                velocity_y, acceleration, steering]


class Extractor3(FeatureExtractor):
    ID = 3
    FEATURES = [
        "Absolute X",
        "Absolute Y",
        "Relative X to target",
        "Relative Y to target",
        "Distance to target",
        "Absolute Rotation (normalized with Cos)",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
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
        absolute_x = parking_lot.car_agent.location.x
        absolute_y = parking_lot.car_agent.location.y
        relative_x = parking_lot.target_park.location.x - parking_lot.car_agent.location.x
        relative_y = parking_lot.target_park.location.y - parking_lot.car_agent.location.y
        absolute_rotation = math.cos(math.radians(parking_lot.car_agent.rotation))
        distance_to_target = parking_lot.car_agent.location.distance_to(parking_lot.target_park.location)
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))
        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

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

        return [absolute_x, absolute_y, relative_x, relative_y, absolute_rotation, distance_to_target,
                relative_rotation,
                angle_to_target,
                velocity_x,
                velocity_y, acceleration, steering, sensor_front, sensor_back, sensor_left, sensor_right]


class Extractor4(FeatureExtractor):
    ID = 4
    FEATURES = [
        "Absolute X (normalized with 1/1200)",
        "Absolute Y (normalized with 1/1200)",
        "Relative X to target (normalized with 1/1200)",
        "Relative Y to target (normalized with 1/1200)",
        "Distance to target (normalized with 1/1200)",
        "Absolute Rotation (normalized with Cos)",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
        "Velocity (normalized with 1/1200)",
        "Acceleration (normalized with 1/1200)",
        "Steering (normalized with 1/100)",
        "Sensor Front (normalized with 1/1200)",
        "Sensor Back (normalized with 1/1200)",
        "Sensor Left (normalized with 1/1200)",
        "Sensor Right (normalized with 1/1200)"
    ]
    input_num = len(FEATURES)

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        factor = 1200

        absolute_x = parking_lot.car_agent.location.x / factor
        absolute_y = parking_lot.car_agent.location.y / factor
        relative_x = (parking_lot.target_park.location.x - parking_lot.car_agent.location.x) / factor
        relative_y = (parking_lot.target_park.location.y - parking_lot.car_agent.location.y) / factor

        absolute_rotation = math.cos(math.radians(parking_lot.car_agent.rotation))
        distance_to_target = (parking_lot.car_agent.location.distance_to(
            parking_lot.target_park.location)) / factor
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))

        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

        velocity = parking_lot.car_agent.velocity.x / factor
        acceleration = parking_lot.car_agent.acceleration / factor
        steering = float(parking_lot.car_agent.steering) / 100
        sensor_front = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.FRONT]) / factor
        sensor_back = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.BACK]) / factor
        sensor_left = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.LEFT]) / factor
        sensor_right = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.RIGHT]) / factor

        return [absolute_x, absolute_y, relative_x, relative_y, absolute_rotation, distance_to_target,
                relative_rotation,
                angle_to_target,
                velocity,
                acceleration, steering, sensor_front, sensor_back, sensor_left, sensor_right]


class Extractor5(FeatureExtractor):
    ID = 5
    FEATURES = [
        "Absolute X (normalized with 1/1200)",
        "Absolute Y (normalized with 1/1200)",
        "Relative X to target (normalized with 1/1200)",
        "Relative Y to target (normalized with 1/1200)",
        "Distance to target (normalized with 1/1200)",
        "Absolute Rotation (normalized with Cos)",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
        "Velocity X (normalized with 1/1200)",
        "Velocity Y (normalized with 1/1200)",
        "Acceleration (normalized with 1/1200)",
        "Steering (normalized with 1/100)",
        "Sensor Front (normalized with 1/1200)",
        "Sensor Back (normalized with 1/1200)",
        "Sensor Left (normalized with 1/1200)",
        "Sensor Right (normalized with 1/1200)"
    ]
    input_num = len(FEATURES)

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        absolute_x = parking_lot.car_agent.location.x / 600
        absolute_y = parking_lot.car_agent.location.y / 600
        relative_x = (parking_lot.target_park.location.x - parking_lot.car_agent.location.x) / 600
        relative_y = (parking_lot.target_park.location.y - parking_lot.car_agent.location.y) / 600

        absolute_rotation = math.cos(math.radians(parking_lot.car_agent.rotation))
        distance_to_target = (parking_lot.car_agent.location.distance_to(
            parking_lot.target_park.location)) / 600
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))

        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

        velocity_x = parking_lot.car_agent.velocity.x / 600
        velocity_y = parking_lot.car_agent.velocity.y / 600
        acceleration = parking_lot.car_agent.acceleration / 600
        steering = float(parking_lot.car_agent.steering) / 50
        sensor_front = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.FRONT]) / 600
        sensor_back = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.BACK]) / 600
        sensor_left = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.LEFT]) / 600
        sensor_right = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.RIGHT]) / 600

        return [absolute_x, absolute_y, relative_x, relative_y, absolute_rotation, distance_to_target,
                relative_rotation,
                angle_to_target,
                velocity_x,
                velocity_y, acceleration, steering, sensor_front, sensor_back, sensor_left, sensor_right]


class ExtractorNew(FeatureExtractor):
    ID = 7
    FEATURES = [
        "Distance to target (normalized with 1/1200)",
        "Absolute Rotation (normalized with Cos)",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
        "Velocity X (normalized with 1/1200)",
        "Velocity Y (normalized with 1/1200)",
        "Acceleration (normalized with 1/1200)",
        "Steering (normalized with 1/100)",
        "All sensors"
    ]
    input_num = 12 + 7

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        factor = parking_lot.car_agent.height
        absolute_rotation = math.cos(math.radians(parking_lot.car_agent.rotation))
        distance_to_target = (parking_lot.car_agent.location.distance_to(
            parking_lot.target_park.location)) / factor
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))

        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

        velocity = parking_lot.car_agent.velocity.magnitude() / factor
        acceleration = parking_lot.car_agent.acceleration / factor
        steering = float(parking_lot.car_agent.steering) / 100
        sensors = [sensor.detect(parking_lot.all_obstacles)[1] for direction in SensorDirection for sensor in
                   parking_lot.car_agent.sensors[direction]]

        return [absolute_rotation, distance_to_target,
                relative_rotation,
                angle_to_target,
                velocity, acceleration, steering, *sensors]


class Extractor6(FeatureExtractor):
    ID = 9
    FEATURES = [
        "Absolute X (normalized with 1/1200)",
        "Absolute Y (normalized with 1/1200)",
        "Front X relative to target (normalized with 1/1200)",
        "Front Y relative to target (normalized with 1/1200)",
        "Relative X to target (normalized with 1/1200)",
        "Relative Y to target (normalized with 1/1200)",
        "Distance from front to target (normalized with 1/1200)",
        "Absolute Rotation (normalized with Cos)",
        "Relative Rotation to target's rotation (normalized with Cos)",
        "Angle to Target (normalized with Cos)",
        "Velocity (normalized with 1/1200)",
        "Acceleration (normalized with 1/1200)",
        "Steering (normalized with 1/100)",
        "Sensor Front (normalized with 1/1200)",
        "Sensor Back (normalized with 1/1200)",
        "Sensor Left (normalized with 1/1200)",
        "Sensor Right (normalized with 1/1200)"
    ]
    input_num = len(FEATURES)

    def get_state(self, parking_lot: ParkingLot) -> List[float]:
        factor = 1200

        absolute_x = parking_lot.car_agent.location.x / factor
        absolute_y = parking_lot.car_agent.location.y / factor
        relative_x = (parking_lot.target_park.location.x - parking_lot.car_agent.location.x) / factor
        relative_y = (parking_lot.target_park.location.y - parking_lot.car_agent.location.y) / factor

        front_x = (parking_lot.target_park.location.x - parking_lot.car_agent.front.x) / factor
        front_y = (parking_lot.target_park.location.y - parking_lot.car_agent.front.y) / factor

        absolute_rotation = math.cos(math.radians(parking_lot.car_agent.rotation))
        distance_to_target = (parking_lot.car_agent.front.distance_to(
            parking_lot.target_park.location)) / factor
        relative_rotation = math.cos(abs(math.radians(parking_lot.car_agent.rotation) - math.radians(
            parking_lot.target_park.rotation)))

        car_front = parking_lot.car_agent.location + pygame.Vector2(
            (parking_lot.car_agent.width / 2 * math.cos(math.radians(parking_lot.car_agent.rotation))),
            (parking_lot.car_agent.width / 2 * math.sin(math.radians(parking_lot.car_agent.rotation + 180))))
        front_vector = car_front - parking_lot.car_agent.location
        to_target_vector = parking_lot.target_park.location - parking_lot.car_agent.location
        angle_to_target = math.cos(math.radians(front_vector.angle_to(to_target_vector)))

        velocity = parking_lot.car_agent.velocity.x / factor
        acceleration = parking_lot.car_agent.acceleration / factor
        steering = float(parking_lot.car_agent.steering) / 100
        sensor_front = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.FRONT]) / factor
        sensor_back = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.BACK]) / factor
        sensor_left = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.LEFT]) / factor
        sensor_right = min(
            sensor.detect(parking_lot.all_obstacles)[1] for sensor in parking_lot.car_agent.sensors[
                SensorDirection.RIGHT]) / factor

        return [absolute_x, absolute_y, relative_x, relative_y, front_x, front_y, absolute_rotation,
                distance_to_target,
                relative_rotation,
                angle_to_target,
                velocity,
                acceleration, steering, sensor_front, sensor_back, sensor_left, sensor_right]

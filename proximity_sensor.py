from enum import Enum
import math

import pygame




class SensorDirection(Enum):
    LEFT = 0
    FRONT = 1
    RIGHT = 2
    BACK = 3


class ProximitySensor:
    def __init__(self, direction: SensorDirection, angle: float, max_distance: float):
        self.direction: SensorDirection = direction
        self.angle = angle
        self.max_distance = max_distance

    def _create_sensor_line(self, car):
        offset = None
        if self.direction == SensorDirection.RIGHT:
            offset = pygame.Vector2((car.height / 2 * math.sin(math.radians(car.rotation))),
                                    (car.height / 2 * math.cos(math.radians(car.rotation))))
        if self.direction == SensorDirection.LEFT:
            offset = pygame.Vector2((car.height / 2 * math.sin(math.radians(car.rotation + 180))),
                                    (car.height / 2 * math.cos(math.radians(car.rotation + 180))))
        if self.direction == SensorDirection.FRONT:
            offset = pygame.Vector2((car.width / 2 * math.cos(math.radians(car.rotation))),
                                    (car.width / 2 * math.sin(math.radians(car.rotation + 180))))
        if self.direction == SensorDirection.BACK:
            offset = pygame.Vector2((car.width / 2 * math.cos(math.radians(car.rotation + 180))),
                                    (car.width / 2 * math.sin(math.radians(car.rotation))))
        start_pos = car.location + offset
        sensor_angle = car.rotation + self.angle
        stop_pos = pygame.Vector2(self.max_distance * math.sin(math.radians(sensor_angle)),
                                  self.max_distance * math.cos(math.radians(sensor_angle)))
        return start_pos, stop_pos

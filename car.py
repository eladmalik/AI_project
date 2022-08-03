from enum import Enum

import pygame
from pygame.math import Vector2
import math

from CarSimSprite import CarSimSprite


class Acceleration(Enum):
    FORWARD = 1
    NEUTRAL = 0
    BACKWARDS = -1


class Steering(Enum):
    LEFT = 1
    NEUTRAL = 0
    RIGHT = -1


TURN_RADIUS = 5
MAX_SPEED = 10
FORWARD_ACCELERATION = 100
BACKWARDS_ACCELERATION = 100


class Car(CarSimSprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str):
        super().__init__(x, y, width, height, rotation, img_path)
        self.velocity = Vector2(0, 0)

    def move(self, acceleration: Acceleration, steering: Steering, time: float):
        a = Vector2(0, 0)
        if acceleration == Acceleration.FORWARD:
            a += Vector2(-(FORWARD_ACCELERATION * math.sin(math.radians(self.rotation))),
                         -(FORWARD_ACCELERATION * math.cos(math.radians(self.rotation))))
        elif acceleration == Acceleration.BACKWARDS:
            a += Vector2((BACKWARDS_ACCELERATION * math.sin(math.radians(self.rotation))),
                         (BACKWARDS_ACCELERATION * math.cos(math.radians(self.rotation))))
        new_location = self.location + self.velocity * time + 0.5 * a * time * time
        new_speed = self.velocity + a * time
        self.velocity = new_speed
        self.update_location(new_location, self.rotation)

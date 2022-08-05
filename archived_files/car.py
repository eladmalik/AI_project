# This is an old attempt to create the car sprite and logic which failed

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


TURN_RADIUS = 50
MAX_SPEED = 10
FORWARD_ACCELERATION = 200
BACKWARDS_ACCELERATION = 200

BASE_VECTOR = Vector2(0, -1)

get_angle = lambda x: x % 360


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

        if steering == Steering.LEFT:
            angular_acc = self.velocity.magnitude_squared() / TURN_RADIUS
            angle = get_angle(self.rotation + 90)
            a += Vector2(-(angular_acc * math.sin(math.radians(angle))),
                         -(angular_acc * math.cos(math.radians(angle))))
        elif steering == Steering.RIGHT:
            angular_acc = self.velocity.magnitude_squared() / TURN_RADIUS
            angle = get_angle(self.rotation - 90)
            a += Vector2(-(angular_acc * math.sin(math.radians(angle))),
                         -(angular_acc * math.cos(math.radians(angle))))
        new_location = self.location + self.velocity * time + 0.5 * a * time * time
        new_speed = self.velocity + a * time
        self.velocity = new_speed
        velocity_direction = get_angle(self.velocity.angle_to(BASE_VECTOR))
        rotation_vector = Vector2(-math.sin(math.radians(self.rotation)),
                                  -math.cos(math.radians(self.rotation)))
        angle_difference = get_angle(self.velocity.angle_to(rotation_vector))
        angle_difference_flipped = get_angle(self.velocity.angle_to(rotation_vector.rotate(180)))
        if angle_difference > angle_difference_flipped:
            velocity_direction = get_angle(velocity_direction + 180)
        # self.rotation = velocity_direction

        self.update_location(new_location, self.rotation)

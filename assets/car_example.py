# This is the (slightly modified) original project which the car's logic was taken from.

import os
from enum import Enum

import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2


class Movement(Enum):
    FORWARD = 1
    NEUTRAL = 0
    BACKWARD = -1
    BREAK = 2


class Steering(Enum):
    LEFT = 1
    NEUTRAL = 0
    RIGHT = -1


class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt: float, movement: Movement, steering: Steering):
        if movement == Movement.FORWARD:
            if self.velocity.x < 0:
                self.acceleration = self.brake_deceleration
            else:
                self.acceleration += 1 * dt
        elif movement == Movement.BACKWARD:
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration -= 1 * dt
        elif movement == Movement.BREAK:
            if abs(self.velocity.x) > dt * self.brake_deceleration:
                self.acceleration = -copysign(self.brake_deceleration, self.velocity.x)
            else:
                self.acceleration = -self.velocity.x / dt
        elif movement == Movement.NEUTRAL:
            if abs(self.velocity.x) > dt * self.free_deceleration:
                self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
            else:
                if dt != 0:
                    self.acceleration = -self.velocity.x / dt
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        if steering == Steering.LEFT:
            self.steering += 30 * dt
        elif steering == Steering.RIGHT:
            self.steering -= 30 * dt
        elif steering == Steering.NEUTRAL:
            self.steering = 0
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))

        # OLD UPDATE FROM HERE
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        car = Car(0, 0)
        ppu = 32

        while not self.exit:
            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()
            movement = Movement.NEUTRAL
            steering = Steering.NEUTRAL
            if pressed[pygame.K_UP]:
                movement = Movement.FORWARD
            elif pressed[pygame.K_DOWN]:
                movement = Movement.BACKWARD
            elif pressed[pygame.K_SPACE]:
                movement = Movement.BREAK

            if pressed[pygame.K_RIGHT]:
                steering = Steering.RIGHT
            elif pressed[pygame.K_LEFT]:
                steering = Steering.LEFT
            # Logic
            car.update(dt, movement, steering)

            # Drawing
            self.screen.fill((0, 0, 0))
            rotated = pygame.transform.rotate(car_image, car.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
            pygame.display.flip()

            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()

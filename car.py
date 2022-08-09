from enum import Enum

from pygame.math import Vector2
from math import sin, radians, degrees, copysign

from CarSimSprite import CarSimSprite
from proximity_sensor import ProximitySensor, SensorDirection


class Movement(Enum):
    """
    Indicates the forward/backward movement of a car
    """
    FORWARD = 1
    NEUTRAL = 0
    BACKWARD = -1
    BRAKE = 2


class Steering(Enum):
    """
    Indicates the side which the steering wheel is rotated to
    """
    LEFT = 1
    NEUTRAL = 0
    RIGHT = -1


MAX_STEERING = 100
MAX_ACCELERATION = 200
MAX_VELOCITY = 150
BRAKE_DECELERATION = 130
FREE_DECELERATION = 15
ACCELERATION_FACTOR = 60
STEERING_FACTOR = 60

MAX_SENSOR_DISTANCE = 80


class Car(CarSimSprite):
    """
    This class represents the car sprite in the simulator.
    The car's logic is based on this tutorial: http://rmgi.blog/pygame-2d-car-tutorial.html
    and its source code: https://github.com/maximryzhov/pygame-car-tutorial
    """

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str,
                 topleft: bool = False):
        super().__init__(x, y, width, height, rotation, img_path, topleft)
        self.velocity = Vector2(0.0, 0.0)
        self.length = width
        self.max_acceleration = MAX_ACCELERATION
        self.max_steering = MAX_STEERING
        self.max_velocity = MAX_VELOCITY
        self.brake_deceleration = BRAKE_DECELERATION
        self.free_deceleration = FREE_DECELERATION
        self.acceleration_factor = ACCELERATION_FACTOR
        self.steering_factor = STEERING_FACTOR
        self.sensors = [ProximitySensor(SensorDirection.FRONT, 0, MAX_SENSOR_DISTANCE)]

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt: float, movement: Movement, steering: Steering):
        """
        updates the movement of the car
        :param dt: the time interval which the car should move
        :param movement: indicates the forward/backward movement of the car
        :param steering: indicates to which side the car should steer
        """
        if movement == Movement.FORWARD:
            if self.velocity.x < 0:
                self.acceleration = self.brake_deceleration
            else:
                self.acceleration += self.acceleration_factor * dt
        elif movement == Movement.BACKWARD:
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration -= self.acceleration_factor * dt
        elif movement == Movement.BRAKE:
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
            self.steering += self.steering_factor * dt
        elif steering == Steering.RIGHT:
            self.steering -= self.steering_factor * dt
        elif steering == Steering.NEUTRAL:
            self.steering = 0
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))

        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.location += self.velocity.rotate(-self.rotation) * dt
        self.rotation += degrees(angular_velocity) * dt

        self.update_location(self.location, self.rotation)

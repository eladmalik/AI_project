from typing import Union
from typing import Tuple

import pygame
from pygame.math import Vector2

from sim.CarSimSprite import CarSimSprite
from sim.car import Car

from sim.assets_images import SOLID_WHITE


class ParkingCell(CarSimSprite):
    """
    This class represents a parking cell in the simulator. a parking cell may hold up to one stationary car.
    """

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 surface: pygame.Surface,
                 car: Car = None, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, surface, topleft)
        assert car is None or isinstance(car, Car), "car argument is not of type \"Car\""

        # since we need the whole parking cell's area to be counted when checking if the agent is in it,
        # we update the mask template to be a solid white rectangle, which is rotated to the cell's angle.
        self.set_mask_template(pygame.transform.scale(SOLID_WHITE, (width, height)))

        self.car = car

    def is_occupied(self) -> bool:
        """
        :return: True iff there is a car in the parking cell.
        """
        return self.car is not None

    def place_car(self, width: float, height: float, img_path: str,
                  location: Union[None, Tuple[float, float], Vector2] = None, rotation=None):
        """
        Places a car in the cell. Should be used before initializing the simulator
        """
        if location is None:
            location = self.rect.center
        if rotation is None:
            rotation = self.rotation
        if not isinstance(location, Vector2):
            location = Vector2(location)
        self.car = Car(location.x, location.y, width, height, rotation, img_path)
        return self

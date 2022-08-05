import os.path

import pygame

from CarSimSprite import CarSimSprite
from car import Car

PATH_SOLID_WHITE = os.path.join("assets", "walter_white.png")


class ParkingCell(CarSimSprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str,
                 car: Car = None, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, img_path, topleft)
        assert car is None or isinstance(car, Car), "car argument is not of type \"Car\""
        self.set_mask_template(pygame.transform.scale(pygame.image.load(PATH_SOLID_WHITE), (width, height)))
        self.car = car

    def is_occupied(self) -> bool:
        return self.car is not None

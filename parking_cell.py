import pygame

from CarSimSprite import CarSimSprite
from car import Car


class ParkingCell(CarSimSprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str,
                 car: Car = None):
        super().__init__(x, y, width, height, rotation, img_path)
        self.car = car

    def is_occupied(self):
        return self.car is not None

import pygame

from CarSimSprite import CarSimSprite


class Car(CarSimSprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str):
        super().__init__(x, y, width, height, rotation, img_path)

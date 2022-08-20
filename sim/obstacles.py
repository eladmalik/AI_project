import pygame

from sim.assets_images import SIDEWALK_IMG
from sim.CarSimSprite import CarSimSprite


class Sidewalk(CarSimSprite):
    DEFAULT_IMAGE = SIDEWALK_IMG

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 surface: pygame.Surface = DEFAULT_IMAGE, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, surface, topleft)

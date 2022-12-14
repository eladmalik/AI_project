import pygame

from assets.assets_images import SIDEWALK_IMG
from simulation.CarSimSprite import CarSimSprite


class Sidewalk(CarSimSprite):
    """
    this class creates the sidewalk sprite
    """
    DEFAULT_IMAGE = SIDEWALK_IMG

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 surface: pygame.Surface = DEFAULT_IMAGE, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, surface, topleft)

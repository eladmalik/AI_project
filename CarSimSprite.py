from typing import Tuple

import pygame


class CarSimSprite(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str):
        """
        This class creates represents a sprite in the simulator
        :param x: the x position of the **CENTER** of the sprite
        :param y: the y position of the **CENTER** of the sprite
        :param width: the width of the sprite in pixels
        :param height: the height of the sprite in pixels
        :param rotation: the angle of the sprite on the board, with 0 pointing upwards (0 points towards
        negative Y). rotation in counter-clockwise (meaning 90 will point towards negative X)
        :param img_path: the path of the sprite's image in the disk
        """
        assert 0 <= rotation < 360, "Rotation must be in range of [0,360) (in degrees)"
        pygame.sprite.Sprite.__init__(self)
        self.rotation = rotation
        self.x = x
        self.y = y

        img = pygame.image.load(img_path)
        self.image_no_rotation = pygame.transform.scale(img, (width, height))
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)

        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)

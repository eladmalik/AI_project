from typing import Tuple, Union
from pygame.math import Vector2

import pygame


class CarSimSprite(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float, img_path: str,
                 topleft: bool = False):
        """
        This class creates represents a sprite in the simulator
        :param x: the x position of the **CENTER** of the sprite
        :param y: the y position of the **CENTER** of the sprite
        :param width: the width of the sprite in pixels
        :param height: the height of the sprite in pixels
        :param rotation: the angle of the sprite on the board, with 0 pointing right (0 points towards
        positive X). rotation in counter-clockwise (meaning 90 will point towards negative Y)
        :param img_path: the path of the sprite's image in the disk
        :param topleft: True if (x,y) states the top left corner of the sprite. False by default.
        """
        # assert 0 <= rotation < 360, "Rotation must be in range of [0,360) (in degrees)"
        pygame.sprite.Sprite.__init__(self)
        self.rotation = rotation
        self.location = Vector2(x, y)
        img = pygame.image.load(img_path)
        self.image_no_rotation = pygame.transform.scale(img, (width, height))
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)
        if topleft:
            self.location = self.image.get_rect(topleft=self.location).center
        self.rect = self.image.get_rect()
        self.rect.center = self.location
        self._mask_template = self.image_no_rotation
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))

    def get_x(self):
        return self.location.x

    def get_y(self):
        return self.location.y

    def get_topleft(self):
        return self.rect.topleft

    def set_mask_template(self, surface_no_rotation: pygame.Surface):
        self._mask_template = surface_no_rotation
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))


    def get_base_dimensions(self):
        return self.image_no_rotation.get_width(), self.image_no_rotation.get_height()

    def get_dimensions(self):
        return self.image.get_width(), self.image.get_height()

    def update_location(self, new_location: Union[Tuple[float, float], Vector2], new_angle: float):
        # assert 0 <= new_angle < 360, "Rotation must be in range of [0,360) (in degrees)"
        if isinstance(new_location, tuple):
            new_location = Vector2(new_location)

        self.location = new_location
        self.rotation = new_angle
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)
        self.rect = self.image.get_rect()
        self.rect.center = self.location
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))

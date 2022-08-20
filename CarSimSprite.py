import math
from typing import Tuple, Union
from pygame.math import Vector2

import pygame


class CarSimSprite(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 surface: pygame.Surface = None,
                 topleft: bool = False):
        """
        This class creates represents a sprite in the simulator
        A sprite is an object which holds an image, dimensions on the screen and location on the screen,
        which usually describes every visible object on the screen. all of its properties may change if
        needed.
        Three main properties the sprite holds are:
        - surface: The image of the sprite (the visual representation of the sprite), scaled and rotated to
        its correct orientation at any given moment. a surface doesn't have a position on the screen. saved in
        the object as self.image (a property of pygame.sprite.Sprite)

        - rectangle (rect): The positional representation of the object on the screen. rect holds both the
        actual dimensions of the sprite and its position on the screen at any given moment (although we
        also chose to save the sprite's position in a dedicated property. This is because most of our
        calculations depend on the center of the sprite but a rect works by default with the top-left
        corner position of the object it describes). rect doesn't hold any information about the image
        of the sprite. saved in the object as self.rect (a property of pygame.sprite.Sprite)

        - mask: The physical representation of the sprite on the screen. Basically a list of all the
        non-transparent pixels of the image. used mainly in order to check for collisions between a
        sprite and other sprites, by checking if one mask overlaps another mask (if pixels in the same
        location on the screen exists in more than one mask). a mask holds no actual location on the
        screen nor dimensions, as it is basically a list of (not necessarily polygon-shaped) pixels. saved
        in the object as self.mask (a property of pygame.sprite.Sprite)

        :param x: the x position of the **CENTER** of the sprite
        :param y: the y position of the **CENTER** of the sprite
        :param width: the width of the sprite in pixels
        :param height: the height of the sprite in pixels
        :param rotation: the angle of the sprite on the board, with 0 pointing right (0 points towards
        positive X). rotation in counter-clockwise (meaning 90 will point towards negative Y)
        :param img_path: the path of the sprite's image in the disk
        :param topleft: True if (x,y) states the top left corner of the sprite. False by default.
        """
        pygame.sprite.Sprite.__init__(self)
        self.rotation: float = rotation
        self.location: Vector2 = Vector2(x, y)
        if surface is not None:
            img = surface
        else:
            img = pygame.Surface((width, height))
        self.width: float = width
        self.height: float = height
        # the base surface of the sprite's image. used in order to calculate the rotated image
        self.image_no_rotation = pygame.transform.scale(img, (width, height))
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)

        if topleft:
            self.location = Vector2(self.image.get_rect(topleft=self.location).center)
        self.rect = self.image.get_rect()
        self.rect.center = self.location

        # used for efficient drawing
        self.prev_rect = self.rect

        # the image template of the mask. this image represents the actual "hitbox" of the object. by
        # default, it's simply the image itself.
        self._mask_template = self.image_no_rotation
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))

        front_off, back_off, left_off, right_off = self.__calculate_offsets()
        self.front = self.location + front_off
        self.back = self.location + back_off
        self.left = self.location + left_off
        self.right = self.location + right_off
        self.frontleft = self.location + front_off + left_off
        self.frontright = self.location + front_off + right_off
        self.backleft = self.location + back_off + left_off
        self.backright = self.location + back_off + right_off

    def __calculate_offsets(self):
        # front is when facing towards the positive side of X
        # right is when facing towards the positive side of Y

        front_offset = pygame.Vector2(
            (self.width / 2 * math.cos(math.radians(self.rotation))),
            (self.width / 2 * math.sin(math.radians(self.rotation + 180))))
        back_offset = pygame.Vector2(
            (self.width / 2 * math.cos(math.radians(self.rotation +
                                                    180))),
            (self.width / 2 * math.sin(math.radians(self.rotation))))
        left_offset = pygame.Vector2(
            (self.height / 2 * math.sin(math.radians(self.rotation + 180))),
            (self.height / 2 * math.cos(math.radians(self.rotation + 180))))
        right_offset = pygame.Vector2(
            (self.height / 2 * math.sin(math.radians(self.rotation))),
            (self.height / 2 * math.cos(math.radians(self.rotation))))
        return front_offset, back_offset, left_offset, right_offset

    def get_x(self):
        return self.location.x

    def get_y(self):
        return self.location.y

    def get_topleft(self):
        return self.rect.topleft

    def get_center(self):
        return self.rect.center

    def set_mask_template(self, surface_no_rotation: pygame.Surface):
        """
        if the desired mask is not the image of the sprite, it can be chnaged to a custom one via this
        function.
        :param surface_no_rotation: the surface (image) of the desired mask, without any rotation.
        """
        self._mask_template = surface_no_rotation
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))

    def set_image(self, surface: pygame.Surface):
        """
        A function which changes the image source of the sprite
        """
        img = surface
        self.image_no_rotation = pygame.transform.scale(img, (self.width, self.height))
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)

    def get_base_dimensions(self):
        """
        :return: the dimension of the sprite, assuming it's not rotated.
        """
        return self.image_no_rotation.get_width(), self.image_no_rotation.get_height()

    def get_dimensions(self):
        """
        :return: the dimension of the rectangle which circumscribes the sprite
        """
        return self.image.get_width(), self.image.get_height()

    def update_location(self, new_location: Union[Tuple[float, float], Vector2], new_angle: float):
        """
        Updates the location of the sprite. also updates its image and mask according to the new rotation
        :param new_location: the new center of the sprite
        :param new_angle: the new angle of the sprite, as described under "rotation" in the constructor's
        documentation.
        """
        self.prev_rect = self.rect
        if isinstance(new_location, tuple):
            new_location = Vector2(new_location)

        self.location = new_location
        self.rotation = new_angle
        self.image = pygame.transform.rotate(self.image_no_rotation, self.rotation)
        self.mask = pygame.mask.from_surface(pygame.transform.rotate(self._mask_template, self.rotation))
        self.rect = self.image.get_rect()
        self.rect.center = self.location

        front_off, back_off, left_off, right_off = self.__calculate_offsets()
        self.front = self.location + front_off
        self.back = self.location + back_off
        self.left = self.location + left_off
        self.right = self.location + right_off
        self.frontleft = self.location + front_off + left_off
        self.frontright = self.location + front_off + right_off
        self.backleft = self.location + back_off + left_off
        self.backright = self.location + back_off + right_off

    def copy(self):
        new_copy = CarSimSprite(self.location.x, self.location.y, self.width, self.height, self.rotation)
        new_copy.image_no_rotation = self.image_no_rotation
        new_copy.image = self.image
        new_copy._mask_template = self._mask_template
        new_copy.mask = self.mask
        new_copy.rect = self.rect
        new_copy.prev_rect = self.prev_rect
        return new_copy

    def __copy__(self):
        return self.copy()

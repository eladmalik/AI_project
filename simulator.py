import os.path
from enum import Enum
from typing import Dict

import pygame

from parking_cell import ParkingCell
from utils import mask_subset_percentage
from parking_lot import ParkingLot

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLOOR = (77, 76, 75)

PATH_AGENT_IMG = os.path.join("assets", "green-car-top-view.png")
PATH_PARKING_IMG = os.path.join("assets", "parking_left.png")
PATH_CAR_IMG = os.path.join("assets", "orange-racing-car-top-view.png")
PATH_BACKGROUND_IMG = os.path.join("assets", "floor.png")


class Results(Enum):
    COLLISION = 1
    AGENT_IN_UNOCCUPIED = 2
    UNOCCUPIED_PERCENTAGE = 3


class Simulator:
    def __init__(self, lot: ParkingLot):
        pygame.init()
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.background = pygame.transform.scale(pygame.image.load(PATH_BACKGROUND_IMG), (self.width,
                                                                                          self.height))
        self.agent = self.parking_lot.car_agent
        self.agent_group = pygame.sprite.Group(self.agent)
        self.stationary_cars_group = pygame.sprite.Group(self.parking_lot.stationary_cars)
        self.parking_cells_group = pygame.sprite.Group(self.parking_lot.parking_cells)
        self.collisionable_group = pygame.sprite.Group(self.parking_lot.stationary_cars)

        pygame.display.set_caption("Car Parking Simulator")
        self.draw_screen()

    def draw_screen(self):
        self.window.fill(FLOOR)
        self.parking_cells_group.draw(self.window)
        self.stationary_cars_group.draw(self.window)
        self.agent_group.draw(self.window)

    def move_agent(self, movement, steering, time):
        self.agent.update(time, movement, steering)
        return {Results.COLLISION: self.is_collision(),
                Results.AGENT_IN_UNOCCUPIED: self.agent_in_unoccupied_cell(),
                Results.UNOCCUPIED_PERCENTAGE: self.get_agent_percentage_in_unoccupied_cells()}

    def is_collision(self):
        agent_rect = self.agent.rect

        # Testing if the agent is in the screen
        window_rect = self.window.get_rect(topleft=(0, 0))
        if not window_rect.contains(agent_rect):
            return True

        # Testing if the agent's rectangle collides with another obstacle's rectangle
        if pygame.sprite.spritecollide(self.agent, self.collisionable_group, False):
            # if there is a collision, we will check the masks for better precision (masks are always
            # inside the object's rectangle, so if the rectangles don't collide the masks won't collide
            # either)
            if True:
                if pygame.sprite.spritecollide(self.agent, self.collisionable_group, False,
                                               pygame.sprite.collide_mask):
                    return True
        # for car in self.parking_lot.stationary_cars:
        #     if agent_rect.colliderect(car.image.get_rect(center=car.location)):
        #         return True

        return False

    def agent_in_unoccupied_cell(self) -> bool:
        for cell in self.parking_lot.get_empty_parking_cells():
            if self.agent.rect.colliderect(cell.rect):
                collision_percentage = mask_subset_percentage(cell, self.agent)
                if collision_percentage >= 1:
                    return True
        return False

    def get_agent_percentage_in_unoccupied_cells(self) -> Dict[ParkingCell, float]:
        collisions = dict()
        for cell in self.parking_lot.get_empty_parking_cells():
            if self.agent.rect.colliderect(cell.rect):
                collisions[cell] = float(mask_subset_percentage(cell, self.agent))
        return collisions

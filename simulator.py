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
PATH_CAR_IMG = os.path.join("assets", "orange-car-top-view.png")
PATH_ICON_IMG = os.path.join("assets", "icon.png")


class Results(Enum):
    """
    keys of values returns by the move function of the simulator. use them in order to gather information
    on the current state of the simulation.
    """
    COLLISION = 1
    AGENT_IN_UNOCCUPIED = 2
    UNOCCUPIED_PERCENTAGE = 3


class Simulator:
    """
    This class is the driver of the simulator. it initializes the pygame framework, takes inputs from the
    agent/player, outputs their outcome and offers the option to draw them to the screen.
    """

    def __init__(self, lot: ParkingLot):
        pygame.init()
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.agent = self.parking_lot.car_agent
        self.agent_group = pygame.sprite.Group(self.agent)
        self.stationary_cars_group = pygame.sprite.Group(self.parking_lot.stationary_cars)
        self.parking_cells_group = pygame.sprite.Group(self.parking_lot.parking_cells)
        self.collisionable_group = pygame.sprite.Group(self.parking_lot.stationary_cars)

        pygame.display.set_caption("Car Parking Simulator")
        pygame.display.set_icon(pygame.image.load(PATH_ICON_IMG))

        self.draw_screen()

    def draw_screen(self):
        """
        A function which draws the current state on the screen. should be called by the main loop after every
        desired object was updated.
        """
        self.window.fill(FLOOR)
        self.parking_cells_group.draw(self.window)
        self.stationary_cars_group.draw(self.window)
        self.agent_group.draw(self.window)

    def move_agent(self, movement, steering, time):
        """
        updates the movement of the car
        :param time: the time interval which the car should move
        :param movement: indicates the forward/backward movement of the car
        :param steering: indicates to which side the car should steer
        :return: A dictionary containing data about the results of the current movement:
                    COLLISION: True/False which indicates if the agent collided with the border or with
                               another object.
                    AGENT_IN_UNOCCUPIED: True/False which indicates if there is a free parking slot which
                                         the agent is completely inside it.
                    UNOCCUPIED_PERCENTAGE: A dictionary holding each free parking cell which the agent has
                                           some part of it inside them. the value of each cell is a
                                           percentage between 0 and 1, indicating how much of the agent is
                                           inside that cell.


        """
        self.agent.update(time, movement, steering)
        return {Results.COLLISION: self.is_collision(),
                Results.AGENT_IN_UNOCCUPIED: self.agent_in_unoccupied_cell(),
                Results.UNOCCUPIED_PERCENTAGE: self.get_agent_percentage_in_unoccupied_cells()}

    def is_collision(self):
        """
        :return: True iff the agent collides with any object which should interrupt the agent (other cars,
        walls, sidewalks, etc.), or the agent goes out of bounds.
        """
        agent_rect = self.agent.rect

        # Testing if the agent is in the screen
        window_rect = self.window.get_rect(topleft=(0, 0))
        if not window_rect.contains(agent_rect):
            return True

        # Testing if the agent's rectangle collides with another obstacle's rectangle
        if pygame.sprite.spritecollide(self.agent, self.collisionable_group, False):
            # if there is a collision between the rectangles of the sprites, we will check the masks for
            # better precision (masks are always inside the object's rectangle, so if the rectangles don't
            # collide the masks won't collide either)
            if pygame.sprite.spritecollide(self.agent, self.collisionable_group, False,
                                           pygame.sprite.collide_mask):
                return True
        return False

    def agent_in_unoccupied_cell(self) -> bool:
        """
        :return: True iff the agent is fully inside a free parking cell.
        """
        for cell in self.parking_lot.free_parking_cells:
            if self.agent.rect.colliderect(cell.rect):
                collision_percentage = mask_subset_percentage(cell, self.agent)
                if collision_percentage >= 1:
                    return True
        return False

    def get_agent_percentage_in_unoccupied_cells(self) -> Dict[ParkingCell, float]:
        """
        :return: A dictionary holding each free parking cell which the agent has some part of it inside
        them. the value of each cell is a percentage between 0 and 1, indicating how much of the agent is
        inside that cell.
        """
        collisions = dict()
        for cell in self.parking_lot.free_parking_cells:
            if self.agent.rect.colliderect(cell.rect):
                collisions[cell] = float(mask_subset_percentage(cell, self.agent))
        return collisions

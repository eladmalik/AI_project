import os.path
from enum import Enum

import pygame

from parking_lot import ParkingLot

WHITE = (255, 255, 255)

PATH_AGENT_IMG = os.path.join("assets", "car.png")
PATH_PARKING_IMG = os.path.join("assets", "gay_gray.png")
PATH_CAR_IMG = os.path.join("assets", "rebel_red.png")


class Results(Enum):
    COLLISION = 1
    AGENT_IN_UNOCCUPIED = 2


class Simulator:
    def __init__(self, lot: ParkingLot):
        pygame.init()
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.agent_group = pygame.sprite.Group(self.parking_lot.car_agent)
        self.stationary_cars_group = pygame.sprite.Group(self.parking_lot.stationary_cars)
        self.parking_cells_group = pygame.sprite.Group(self.parking_lot.parking_cells)

        pygame.display.set_caption("Car Parking Simulator")
        self.draw_screen()

    def draw_screen(self):
        self.window.fill(WHITE)
        self.parking_cells_group.draw(self.window)
        self.stationary_cars_group.draw(self.window)
        self.agent_group.draw(self.window)

    def move_agent(self, movement, steering, time):
        self.parking_lot.car_agent.update(time, movement, steering)
        return {Results.COLLISION: self.is_collision(),
                Results.AGENT_IN_UNOCCUPIED: self.agent_in_unoccupied_cell()}

    def is_collision(self):
        agent_rect = self.parking_lot.car_agent.image.get_rect(center=self.parking_lot.car_agent.location)

        # Testing if the agent is in the screen
        window_rect = self.window.get_rect(topleft=(0, 0))
        if not window_rect.contains(agent_rect):
            return True

        # Testing if the agent collides with another car
        for car in self.parking_lot.stationary_cars:
            if agent_rect.colliderect(car.image.get_rect(center=car.location)):
                return True

        return False

    def agent_in_unoccupied_cell(self):
        agent_rect = self.parking_lot.car_agent.image.get_rect(center=self.parking_lot.car_agent.location)
        for cell in self.parking_lot.get_empty_parking_cells():
            if cell.image.get_rect(center=cell.location).contains(agent_rect):
                return True
        return False

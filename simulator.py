import os.path

import pygame

from parking_lot import ParkingLot

WHITE = (255, 255, 255)

PATH_AGENT_IMG = os.path.join("assets", "car.png")
PATH_PARKING_IMG = os.path.join("assets", "gay_gray.png")
PATH_CAR_IMG = os.path.join("assets", "rebel_red.png")


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

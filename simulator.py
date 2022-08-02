import os.path

import pygame
from utils import blit_rotate_center

from parking_lot import ParkingLot

PURE_WHITE = (255, 255, 255)
GAY_GRAY = (92, 92, 92)
REBEL_RED = (115, 5, 12)
# GENDERLESS_GREEN = (139, 214, 111)

PATH_AGENT_IMG = os.path.join("assets", "genderless_green.png")


class Simulator:
    def __init__(self, lot: ParkingLot):
        pygame.init()
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.object_map = dict()
        pygame.display.set_caption("Car Parking Simulator")
        self.create_objects()
        self.draw_screen()

    def create_objects(self):
        agent_img = pygame.image.load(PATH_AGENT_IMG)
        agent_surface = pygame.transform.scale(agent_img,
                                               (self.parking_lot.car_agent.width,
                                                self.parking_lot.car_agent.height))

        self.object_map[self.parking_lot.car_agent] = agent_surface
        for cell in self.parking_lot.parking_cells:
            rect = pygame.Surface((cell.width, cell.height))
            rect.fill(GAY_GRAY)
            self.object_map[cell] = rect
            if cell.is_occupied():
                rect = pygame.Surface((cell.car.width, cell.car.height))
                rect.fill(REBEL_RED)
                self.object_map[cell.car] = rect

    def draw_screen(self):
        self.window.fill(PURE_WHITE)
        for cell in self.parking_lot.parking_cells:
            self.window.blit(self.object_map[cell], (cell.x, cell.y))
            if cell.is_occupied():
                self.window.blit(self.object_map[cell.car], (cell.car.x, cell.car.y))

        blit_rotate_center(self.window, self.object_map[self.parking_lot.car_agent],
                           (self.parking_lot.car_agent.x, self.parking_lot.car_agent.y),
                           self.parking_lot.car_agent.rotation)

        # self.window.blit(self.object_map[self.parking_lot.car_agent], (self.parking_lot.car_agent.x,
        #                                                                self.parking_lot.car_agent.y))

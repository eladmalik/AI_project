import os.path

import pygame

from parking_lot import ParkingLot

PURE_WHITE = (255, 255, 255)
GAY_GRAY = (92, 92, 92)
REBEL_RED = (115, 5, 12)
GENDERLESS_GREEN = (139, 214, 111)

PATH_AGENT_IMG = os.path.join("assets", "genderless_green.png")
PATH_PARKING_IMG = os.path.join("assets", "gay_gray.png")
PATH_CAR_IMG = os.path.join("assets", "rebel_red.png")


class Simulator:
    def __init__(self, lot: ParkingLot):
        pygame.init()
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.sprites_group = pygame.sprite.Group()
        self.sprites_group.add(self.parking_lot.get_all_sprites())
        pygame.display.set_caption("Car Parking Simulator")
        self.draw_screen()

    def draw_screen(self):
        self.window.fill(PURE_WHITE)
        self.sprites_group.draw(self.window)

    def move_agent(self, acceleration, steering, time):
        self.parking_lot.car_agent.move(acceleration, steering, time)

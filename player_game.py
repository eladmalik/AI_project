import sys

import pygame

from simulator import Simulator
from parking_lot import ParkingLot
from parking_cell import ParkingCell
from car import Car

if __name__ == '__main__':
    car = Car(100, 100, 100, 50, 0)
    cell1 = ParkingCell(90, 80, 200, 200, 0, car)
    cell2 = ParkingCell(300, 80, 200, 200, 0)
    agent_car = Car(600, 500, 50, 100, 30)
    lot = ParkingLot(1000, 1000, agent_car, [cell1, cell2])
    sim = Simulator(lot)
    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

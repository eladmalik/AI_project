import sys

import pygame

from simulator import Simulator, PATH_PARKING_IMG, PATH_AGENT_IMG, PATH_CAR_IMG
from parking_lot import ParkingLot
from parking_cell import ParkingCell
from car import Car

if __name__ == '__main__':
    car = Car(100, 100, 100, 50, 0, PATH_CAR_IMG)
    cell1 = ParkingCell(100, 100, 200, 200, 0, PATH_PARKING_IMG, car)
    cell2 = ParkingCell(300, 300, 200, 200, 0, PATH_PARKING_IMG)
    agent_car = Car(600, 500, 50, 100, 30, PATH_AGENT_IMG)
    lot = ParkingLot(1000, 1000, agent_car, [cell1, cell2])
    sim = Simulator(lot)
    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

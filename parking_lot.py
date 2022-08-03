from typing import Iterable, List

import pygame.sprite

from parking_cell import ParkingCell
from car import Car


class ParkingLot:
    def __init__(self, width: int, height: int, agent: Car, parking_cells: List[ParkingCell]):
        self.width: int = width
        self.height: int = height
        self.car_agent: Car = agent
        self.parking_cells: List[ParkingCell] = parking_cells
        self.stationary_cars = [cell.car for cell in self.parking_cells if cell.is_occupied()]

    def get_empty_parking_cells(self) -> Iterable[ParkingCell]:
        empty_cells = []
        for cell in self.parking_cells:
            if not cell.is_occupied():
                empty_cells.append(cell)
        return empty_cells

    def get_all_sprites(self) -> Iterable[pygame.sprite.Sprite]:
        all_sprites = [self.car_agent]
        for cell in self.parking_cells:
            all_sprites.append(cell)
            if cell.is_occupied():
                all_sprites.append(cell.car)
        return all_sprites

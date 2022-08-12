from typing import Iterable, List

import pygame.sprite

from CarSimSprite import CarSimSprite
from parking_cell import ParkingCell
from car import Car


class ParkingLot:
    """
    A class which packs every object which should be inside the simulated parking lot.
    """

    def __init__(self, width: int, height: int, agent: Car, parking_cells: List[ParkingCell], obstacles:
    List[CarSimSprite], target_park: ParkingCell = None):
        self.width: int = width
        self.height: int = height
        self.car_agent: Car = agent
        self.parking_cells: List[ParkingCell] = parking_cells
        self.target_park = target_park
        if self.target_park is not None:
            assert not self.target_park.is_occupied()
        self.obstacles: List[CarSimSprite] = obstacles  # excluding the stationary cars
        self.stationary_cars = [cell.car for cell in self.parking_cells if cell.is_occupied()]
        self.free_parking_cells = [cell for cell in self.parking_cells if not cell.is_occupied()]
        self.all_obstacles: List[CarSimSprite] = self.obstacles + self.stationary_cars

    def get_all_sprites(self) -> Iterable[pygame.sprite.Sprite]:
        """
        :return: every sprite that is in the parking lot
        """
        all_sprites = [self.car_agent]
        for cell in self.parking_cells:
            all_sprites.append(cell)
            if cell.is_occupied():
                all_sprites.append(cell.car)
        return all_sprites

from typing import Iterable

from parking_cell import ParkingCell
from car import Car


class ParkingLot:
    def __init__(self, width: int, height: int, agent: Car, parking_cells: Iterable[ParkingCell]):
        self.width: int = width
        self.height: int = height
        self.car_agent: Car = agent
        self.parking_cells: Iterable[ParkingCell] = parking_cells

    def get_empty_parking_cells(self):
        empty_cells = set()
        for cell in self.parking_cells:
            if not cell.is_occupied():
                empty_cells.add(cell)
        return empty_cells

from enum import Enum
import random
from typing import Tuple
import math

from car import Car
from obstacles import Sidewalk
from parking_cell import ParkingCell
from parking_lot import ParkingLot
from assets_paths import PATH_AGENT_IMG, PATH_PARKING_SIDEWALK_IMG


class ParkingType(Enum):
    PARALLEL = 0
    VERTICAL = 1


def create_parallel_parking_cells(screen_size, sidewalk_width, side, car_size: Tuple[float, float]):
    width_scale = int(round(1.4 + 0.7 * random.random()))
    height_scale = int(round(1.35 + 2.1 * random.random()))
    width = car_size[0] * width_scale
    height = car_size[1] * height_scale
    num_of_cells = int(math.floor(screen_size / width))
    startpos = (0, 0)
    rotation = 0
    if side == 0:
        startpos = (sidewalk_width, 0)
        rotation = 90
    elif side == 1:
        startpos = (0, sidewalk_width)
    elif side == 2:
        startpos = (screen_size - (sidewalk_width + height), 0)
        rotation = 270
    elif side == 3:
        startpos = (0, screen_size - (sidewalk_width + height))
        rotation = 180
    parking_cells = []
    current_position = [startpos[0], startpos[1]]
    for i in range(num_of_cells):
        parking_cells.append(ParkingCell(current_position[0], current_position[1], width, height, rotation,
                                         PATH_PARKING_SIDEWALK_IMG, topleft=True))
        if side % 2 == 0:
            current_position[1] += width
        else:
            current_position[0] += width
    return parking_cells


def scenario1_parallel():
    car_size = (100, 50)
    screen_size = random.randint(700, 1200)
    num_of_sidewalks = random.randint(1, 2)
    sidewalk_corner = random.randint(0, 3)
    sidewalk_scale = 0.1 + 0.2 * random.random()
    sidewalks = []
    sidewalk_width = screen_size * sidewalk_scale
    parking_cells = []
    for _ in range(num_of_sidewalks):
        if sidewalk_corner == 0:
            # left sidewalk
            sidewalks.append(Sidewalk(0, 0, sidewalk_width, screen_size, 0, topleft=True))
        elif sidewalk_corner == 1:
            # top sidewalk
            sidewalks.append(Sidewalk(0, 0, screen_size, sidewalk_width, 0, topleft=True))
        elif sidewalk_corner == 2:
            # right sidewalk
            sidewalks.append(
                Sidewalk(screen_size - sidewalk_width, 0, sidewalk_width, screen_size, 0, topleft=True))
        elif sidewalk_corner == 3:
            # down sidewalk
            sidewalks.append(Sidewalk(0, screen_size - sidewalk_width, screen_size, sidewalk_width, 0,
                                      topleft=True))
        parking_cells += create_parallel_parking_cells(screen_size, sidewalk_width, sidewalk_corner,
                                                       car_size)
        sidewalk_corner += 2
        sidewalk_corner %= 4
    sidewalk_corner += 2
    sidewalk_corner %= 4

    return ParkingLot(screen_size, screen_size, Car(screen_size // 2, screen_size // 2, 100, 50, 0,
                                                    PATH_AGENT_IMG), parking_cells, sidewalks)


def generate_lot():
    parking_type = random.choice([park_type for park_type in ParkingType])
    if parking_type == ParkingType.PARALLEL:
        pass

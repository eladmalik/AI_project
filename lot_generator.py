from enum import Enum
import random
from typing import Tuple, List
import math

import pygame.sprite

from car import Car
from obstacles import Sidewalk
from parking_cell import ParkingCell
from parking_lot import ParkingLot
from assets_paths import PATH_AGENT_IMG, PATH_PARKING_SIDEWALK_IMG, PATH_PARKING_SIDEWALK_TARGET_IMG, \
    PATH_CAR_IMG


class ParkingType(Enum):
    PARALLEL = 0
    VERTICAL = 1


def create_vertical_parking_cells(screen_size, sidewalk_width, side, car_size: Tuple[float, float]) -> List[ParkingCell]:
    height_scale = int(round(1.5 + 0.7 * random.random()))
    width_scale = int(round(1.4 + 2.1 * random.random()))
    width = car_size[1] * width_scale
    height = car_size[0] * height_scale
    num_of_cells = int(math.floor(screen_size / width))
    start_pos = (0, 0)
    rotation = 0
    max_offset = int(round((screen_size - (width * num_of_cells))))
    offset = random.randint(0, max_offset - 1)
    if side == 0:
        start_pos = (sidewalk_width, offset)
        rotation = 90
    elif side == 1:
        start_pos = (offset, sidewalk_width)
    elif side == 2:
        start_pos = (screen_size - (sidewalk_width + height), offset)
        rotation = 270
    elif side == 3:
        start_pos = (offset, screen_size - (sidewalk_width + height))
        rotation = 180
    parking_cells = []
    current_position = [start_pos[0], start_pos[1]]
    for i in range(num_of_cells):
        parking_cells.append(ParkingCell(current_position[0], current_position[1], width, height, rotation,
                                         PATH_PARKING_SIDEWALK_IMG, topleft=True))
        if side % 2 == 0:
            current_position[1] += width
        else:
            current_position[0] += width
    return parking_cells


def create_parallel_parking_cells(screen_size, sidewalk_width, side, car_size: Tuple[float, float]) -> List[ParkingCell]:
    width_scale = int(round(1.5 + 0.7 * random.random()))
    height_scale = int(round(1.4 + 2.1 * random.random()))
    width = car_size[0] * width_scale
    height = car_size[1] * height_scale
    num_of_cells = int(math.floor(screen_size / width))
    start_pos = (0, 0)
    rotation = 0
    max_offset = int(round((screen_size - (width * num_of_cells))))
    offset = random.randint(0, max_offset - 1)
    if side == 0:
        start_pos = (sidewalk_width, offset)
        rotation = 90
    elif side == 1:
        start_pos = (offset, sidewalk_width)
    elif side == 2:
        start_pos = (screen_size - (sidewalk_width + height), offset)
        rotation = 270
    elif side == 3:
        start_pos = (offset, screen_size - (sidewalk_width + height))
        rotation = 180
    parking_cells = []
    current_position = [start_pos[0], start_pos[1]]
    for i in range(num_of_cells):
        parking_cells.append(ParkingCell(current_position[0], current_position[1], width, height, rotation,
                                         PATH_PARKING_SIDEWALK_IMG, topleft=True))
        if side % 2 == 0:
            current_position[1] += width
        else:
            current_position[0] += width
    return parking_cells


def scenario1_parallel() -> ParkingLot:
    car_size = (100, 50)
    screen_size = random.randint(700, 1200)
    num_of_sidewalks = random.randint(1, 2)
    sidewalk_corner = random.randint(0, 3)
    sidewalk_scale = 0.1 + 0.13 * random.random()
    sidewalks = []
    sidewalk_width = screen_size * sidewalk_scale
    parking_cells: List[ParkingCell] = []
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

    # choosing parking target
    target_index = random.randint(0, len(parking_cells) - 1)
    parking_cells[target_index].set_image(PATH_PARKING_SIDEWALK_TARGET_IMG)
    for i in range(len(parking_cells)):
        if i != target_index and random.choice([True, False]):
            parking_cells[i].place_car(car_size[0], car_size[1], PATH_CAR_IMG, rotation=random.choice([
                parking_cells[i].rotation, parking_cells[i].rotation + 180]))

    agent_offset_x = random.randint(-screen_size // 15, screen_size // 15)
    agent_offset_y = random.randint(-screen_size // 15, screen_size // 15)
    agent = Car((screen_size // 2) + agent_offset_x, (screen_size // 2) + agent_offset_y, car_size[0],
                car_size[1], random.random() * 360, PATH_AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, parking_cells, sidewalks)


def scenario1_perpendicular() -> ParkingLot:
    car_size = (100, 50)
    screen_size = random.randint(700, 1200)
    num_of_sidewalks = random.randint(1, 2)
    sidewalk_corner = random.randint(0, 3)
    sidewalk_scale = 0.1 + 0.13 * random.random()
    sidewalks = []
    sidewalk_width = screen_size * sidewalk_scale
    parking_cells: List[ParkingCell] = []
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
        parking_cells += create_vertical_parking_cells(screen_size, sidewalk_width, sidewalk_corner,
                                                       car_size)
        sidewalk_corner += 2
        sidewalk_corner %= 4
    sidewalk_corner += 2
    sidewalk_corner %= 4

    # choosing parking target
    target_index = random.randint(0, len(parking_cells) - 1)
    parking_cells[target_index].set_image(PATH_PARKING_SIDEWALK_TARGET_IMG)
    for i in range(len(parking_cells)):
        if i != target_index and random.choice([True, False]):
            parking_cells[i].place_car(car_size[0], car_size[1], PATH_CAR_IMG, rotation=random.choice([
                parking_cells[i].rotation + 270, parking_cells[i].rotation + 90]))

    agent_offset_x = random.randint(-screen_size // 15, screen_size // 15)
    agent_offset_y = random.randint(-screen_size // 15, screen_size // 15)
    agent = Car((screen_size // 2) + agent_offset_x, (screen_size // 2) + agent_offset_y, car_size[0],
                car_size[1], random.random() * 360, PATH_AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, parking_cells, sidewalks)


def generate_lot():
    parking_type = random.choice([park_type for park_type in ParkingType])
    lot = None
    if parking_type == ParkingType.PARALLEL:
        lot = scenario1_parallel()
        while pygame.sprite.spritecollide(lot.car_agent, pygame.sprite.Group(lot.obstacles,
                                                                             lot.stationary_cars), False,
                                          pygame.sprite.collide_mask):
            lot = scenario1_parallel()

    elif parking_type == ParkingType.VERTICAL:
        # right now same as parallel, should be changed after creating the vertical template
        lot = scenario1_perpendicular()
        while pygame.sprite.spritecollide(lot.car_agent, pygame.sprite.Group(lot.obstacles,
                                                                             lot.stationary_cars), False,
                                          pygame.sprite.collide_mask):
            lot = scenario1_perpendicular()
    return lot

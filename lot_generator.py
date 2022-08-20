from enum import Enum
import random
from typing import Tuple, List, Callable
import math

import pygame.sprite

from car import Car
from obstacles import Sidewalk
from parking_cell import ParkingCell
from parking_lot import ParkingLot
from assets_images import AGENT_IMG, PARKING_SIDEWALK_IMG, PARKING_SIDEWALK_TARGET_IMG, \
    CAR_IMG, PARKING_IMG, PARKING_TARGET_IMG

LotGenerator = Callable[[], ParkingLot]


class ParkingType(Enum):
    PARALLEL = 0
    VERTICAL = 1


def create_vertical_parking_cells_old(screen_size, sidewalk_width, side, car_size: Tuple[float, float]) -> \
        List[
            ParkingCell]:
    height_scale = 1.5 + (0.7 * random.random())
    width_scale = 1.4 + (2.1 * random.random())
    width = round(car_size[1] * width_scale)
    height = round(car_size[0] * height_scale)
    num_of_cells = int(math.floor(screen_size / width))
    start_pos = (0, 0)
    rotation = 0
    max_offset = int(round((screen_size - (width * num_of_cells))))
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
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
                                         PARKING_SIDEWALK_IMG, topleft=True))
        if side % 2 == 0:
            current_position[1] += width
        else:
            current_position[0] += width
    return parking_cells


def create_vertical_parking_cells(screen_size, sidewalk_width, side, car_size: Tuple[float, float]) -> List[
    ParkingCell]:
    width_scale = 1.5 + (0.7 * random.random())
    height_scale = 1.4 + (2.1 * random.random())
    width = round(car_size[0] * width_scale)
    height = round(car_size[1] * height_scale)
    num_of_cells = int(math.floor(screen_size / height))
    start_pos = (0, 0)
    rotation = 0
    max_offset = int(round((screen_size - (height * num_of_cells))))
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    if side == 0:
        start_pos = (sidewalk_width, offset)
        rotation = 0
    elif side == 1:
        start_pos = (offset, sidewalk_width)
        rotation = 270
    elif side == 2:
        start_pos = (screen_size - (sidewalk_width + width), offset)
        rotation = 180
    elif side == 3:
        start_pos = (offset, screen_size - (sidewalk_width + width))
        rotation = 90
    parking_cells = []
    current_position = [start_pos[0], start_pos[1]]
    for i in range(num_of_cells):
        parking_cells.append(ParkingCell(current_position[0], current_position[1], width, height, rotation,
                                         PARKING_IMG, topleft=True))
        if side % 2 == 0:
            current_position[1] += height
        else:
            current_position[0] += height
    return parking_cells


def create_parallel_parking_cells(screen_size, sidewalk_width, side, car_size: Tuple[float, float]) -> List[
    ParkingCell]:
    width_scale = 1.5 + (0.7 * random.random())
    height_scale = 1.4 + (2.1 * random.random())
    width = round(car_size[0] * width_scale)
    height = round(car_size[1] * height_scale)
    num_of_cells = int(math.floor(screen_size / width))
    start_pos = (0, 0)
    rotation = 0
    max_offset = int(round((screen_size - (width * num_of_cells))))
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
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
                                         PARKING_SIDEWALK_IMG, topleft=True))
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
    parking_cells[target_index].set_image(PARKING_SIDEWALK_TARGET_IMG)
    for i in range(len(parking_cells)):
        if i != target_index and random.choice([True, False]):
            rotation = random.choice([parking_cells[i].rotation, parking_cells[i].rotation + 180])
            trash_parking = random.random()
            if trash_parking > 0.75:
                rotation += -20 + 40 * random.random()
            parking_cells[i].place_car(car_size[0], car_size[1], CAR_IMG, rotation=rotation)

    agent_offset_x = random.randint(-screen_size // 15, screen_size // 15)
    agent_offset_y = random.randint(-screen_size // 15, screen_size // 15)
    agent = Car((screen_size // 2) + agent_offset_x, (screen_size // 2) + agent_offset_y, car_size[0],
                car_size[1], random.random() * 360, AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, parking_cells, sidewalks, parking_cells[target_index])


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
    parking_cells[target_index].set_image(PARKING_TARGET_IMG)
    for i in range(len(parking_cells)):
        if i != target_index and random.choice([True, False]):
            rotation = random.choice([parking_cells[i].rotation, parking_cells[i].rotation + 180])
            trash_parking = random.random()
            if trash_parking > 0.75:
                rotation += -20 + 40 * random.random()
            parking_cells[i].place_car(car_size[0], car_size[1], CAR_IMG, rotation=rotation)

    agent_offset_x = random.randint(-screen_size // 15, screen_size // 15)
    agent_offset_y = random.randint(-screen_size // 15, screen_size // 15)
    agent = Car((screen_size // 2) + agent_offset_x, (screen_size // 2) + agent_offset_y, car_size[0],
                car_size[1], random.random() * 360, AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, parking_cells, sidewalks, parking_cells[target_index])


def generate_lot() -> ParkingLot:
    # parking_type = random.choice([park_type for park_type in ParkingType])
    parking_type = ParkingType.VERTICAL
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


def example0() -> ParkingLot:
    screen_size = 1000
    parking_cell = ParkingCell(600, 300, 150, 75, 180, PARKING_TARGET_IMG, topleft=True)
    agent = Car(200, 700, 100, 50, 90, AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, [parking_cell], [], parking_cell)


def example1() -> ParkingLot:
    sidewalk_left = Sidewalk(900, 0, 100, 1000, 0, topleft=True)
    sidewalk_right = Sidewalk(0, 0, 100, 1000, 0, topleft=True)
    offset_x = -30 + 60 * random.random()
    offset_y = -30 + 60 * random.random()
    agent = Car(500 + offset_x, 500 + offset_y, 100, 50, random.random() * 360, AGENT_IMG)
    parking_cells = [
        ParkingCell(100, 0, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 130, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 260, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 390, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                         CAR_IMG),
        ParkingCell(100, 520, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 650, 130, 65, 90, PARKING_SIDEWALK_IMG, topleft=True),

        ParkingCell(835, 0, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 130, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 260, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                          CAR_IMG,
                                                                                          rotation=50),
        ParkingCell(835, 390, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 520, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                          CAR_IMG,
                                                                                          rotation=70),
        ParkingCell(835, 650, 130, 65, 270, PARKING_SIDEWALK_TARGET_IMG, topleft=True),
    ]
    ParkingCell(800, 0, 130, 65, 270, PARKING_SIDEWALK_IMG, topleft=True)
    lot = ParkingLot(1000, 1000, agent, parking_cells, [sidewalk_left, sidewalk_right], parking_cells[-1])
    return lot


def _get_random_place_inside_board(width, height, board_size):
    board_rect = pygame.Rect(0, 0, board_size, board_size)
    rotation = random.random() * 360
    surface = pygame.transform.rotate(pygame.Surface((width, height)), rotation)
    x, y = random.randint(width, board_size - width), random.randint(height, board_size - height)
    rect = surface.get_rect(center=(x, y))
    while not board_rect.contains(rect):
        rotation = random.random() * 360
        surface = pygame.transform.rotate(pygame.Surface((width, height)), rotation)
        x, y = random.randint(width, board_size - width), random.randint(height, board_size - height)
        rect = surface.get_rect(center=(x, y))
    return x, y, rotation


def generate_only_target() -> ParkingLot:
    car_size = (100, 50)
    screen_size = random.randint(700, 1200)
    parking_scale = 1.6
    parking_width = car_size[0] * parking_scale
    parking_height = car_size[1] * parking_scale
    parking_x, parking_y, parking_rotation = _get_random_place_inside_board(parking_width, parking_height,
                                                                            screen_size)
    parking_cell = ParkingCell(parking_x, parking_y, parking_width, parking_height, parking_rotation,
                               PARKING_TARGET_IMG)
    car_x, car_y, car_rotation = _get_random_place_inside_board(car_size[0], car_size[1], screen_size)
    agent = Car(car_x, car_y, car_size[0], car_size[1], car_rotation, AGENT_IMG)
    return ParkingLot(screen_size, screen_size, agent, [parking_cell], [], parking_cell)

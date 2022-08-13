import math
import os.path
import sys
from enum import Enum
from typing import Dict, Union

import pygame

from CarSimSprite import CarSimSprite
from car import Car
from feature_extractor import FeatureExtractor
from parking_cell import ParkingCell
from utils import mask_subset_percentage
from parking_lot import ParkingLot
from reward_analyzer import RewardAnalyzer, Results

from assets_paths import PATH_AGENT_IMG, PATH_PARKING_IMG, PATH_PARKING_SIDEWALK_IMG, PATH_CAR_IMG, \
    PATH_ICON_IMG, PATH_FLOOR_IMG

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLOOR = (77, 76, 75)


class DrawingMethod(Enum):
    FULL = 0
    SKIP_OBSTACLES = 1
    BACKGROUND_SNAPSHOT = 2


class Simulator:
    """
    This class is the driver of the simulator. it initializes the pygame framework, takes inputs from the
    agent/player, outputs their outcome and offers the option to draw them to the screen.
    """

    def __init__(self, lot: ParkingLot, reward_analyzer: RewardAnalyzer, feature_extractor: FeatureExtractor,
                 background_image: Union[str, None] = None,
                 drawing_method=False,
                 full_refresh_rate: int = 30):
        """
        The constructor of the simulator
        :param lot: The parking lot

        :param reward_analyzer: the analyzer responsible for calculating the reward for the current state

        :param background_image: setting this to an image's path will make the simulator to draw the image
        as a background. keeping it as None will make the simulator to draw the default background color.

        :param drawing_method: The method used to draw the screen:
               DrawingMethod.FULL: draws every object every screen update.

               DrawingMethod.SKIP_OBSTACLES: draws only the background image, the parking cells and the
               agent every screen update. since the agent shouldn't run over obstacles (the simulator
               should terminate the simulation when the agent has collided), the agent shouldn't be drawn
               over these objects at all.
               note: since we are drawing the whole agent's rectangle to the
               screen and not just his mask, the screen might look a little wrong when the agent is close
               to obstacles. this is just a visual de-sync and should not harm functionality at all.

               Drawing_method.BACKGROUND_SNAPSHOT: saves a copy of the screen without the agent,
               and every update it replaces the previous agent's rectangle with the portion of the copy
               image which is in the agent's position

        :param full_refresh_rate: if efficient_drawing is set to True, the screen will be drawn completely
               once every this number of iterations. higher values means more efficient drawing,
               but less nice-looking screen
        """
        pygame.init()
        self.iteration_counter: int = 0
        self.full_refresh_rate: int = full_refresh_rate
        self.parking_lot: ParkingLot = lot
        self.width: float = self.parking_lot.width
        self.height: float = self.parking_lot.height
        self.window: pygame.Surface = pygame.display.set_mode((self.width, self.height))
        self.agent: Car = self.parking_lot.car_agent
        self.agent_group: pygame.sprite.Group = pygame.sprite.Group(self.agent)
        self.stationary_cars_group: pygame.sprite.Group = pygame.sprite.Group(
            self.parking_lot.stationary_cars)
        self.parking_cells_group: pygame.sprite.Group = pygame.sprite.Group(self.parking_lot.parking_cells)
        self.reward_analyzer: RewardAnalyzer = reward_analyzer
        self.feature_extractor: FeatureExtractor = feature_extractor
        borders = [
            CarSimSprite(-1, 0, 2, self.height, 0, topleft=True),
            CarSimSprite(self.width, 0, 2, self.height, 0, topleft=True),
            CarSimSprite(0, -1, self.width, 2, 0, topleft=True),
            CarSimSprite(0, self.height, self.width, 2, 0, topleft=True)
        ]
        self.obstacles_group: pygame.sprite.Group = pygame.sprite.Group(self.parking_lot.all_obstacles,
                                                                        borders)

        self.background_img = None
        if background_image is not None:
            self.background_img = pygame.transform.scale(pygame.image.load(background_image),
                                                         (lot.width, lot.height))
        self.bg_snapshot = self._create_background_snapshot()

        pygame.display.set_caption("Car Parking Simulator")
        pygame.display.set_icon(pygame.image.load(PATH_ICON_IMG))

        self._drawing_method = drawing_method

        self._draw_screen_full()
        pygame.display.update()

    def _create_background_snapshot(self):
        """
        Creates a surface which contains the background and every sprite in the simulator, execpt for the
        agent.
        :return the surface
        """
        if self.background_img is not None:
            snapshot = self.background_img.copy()
        else:
            snapshot = pygame.Surface((self.width, self.height))
            snapshot.fill(FLOOR)
        self.parking_cells_group.draw(snapshot)
        self.obstacles_group.draw(snapshot)
        return snapshot

    def _draw_screen_full(self):
        """
        A function which draws all the objects on the screen in the correct order
        """
        # deleting the whole screen and filling it with the floor's color or the background image
        if self.background_img is not None:
            self.window.blit(self.background_img, (0, 0))
        else:
            self.window.fill(FLOOR)
        # drawing the parking cells
        self.parking_cells_group.draw(self.window)
        # drawing the obstacles
        self.obstacles_group.draw(self.window)
        # drawing the agent
        self.agent_group.draw(self.window)

    def _draw_screen_no_obstacles(self):
        """
        A function which draws all the objects on the screen in the correct order, but skips the obstacles
        and redraws the background only around the agent
        """
        # deleting the previous car
        if self.background_img is not None:
            sub_rect = self.agent.prev_rect.clip(self.window.get_rect(topleft=(0, 0)))
            if sub_rect.size != (0, 0):
                subsurface = self.background_img.subsurface(sub_rect)
                self.window.blit(subsurface, sub_rect.topleft)
        else:
            pygame.draw.rect(self.window, FLOOR, self.agent.prev_rect)
        # drawing the parking cells
        self.parking_cells_group.draw(self.window)
        # not drawing the obstacles - hence allowing for faster refresh rate.

        # drawing the agent
        self.agent_group.draw(self.window)

    def _draw_screen_snapshot(self):
        """
        A function which redraws the screen only around the agent, using the snapshot image.
        """
        # removing the previous drawing of the agent
        sub_rect = self.agent.prev_rect.clip(self.window.get_rect(topleft=(0, 0)))
        if sub_rect.size != (0, 0):
            subsurface = self.bg_snapshot.subsurface(sub_rect)
            self.window.blit(subsurface, sub_rect.topleft)
        # drawing the agent in its new position
        self.agent_group.draw(self.window)

    def update_screen(self):
        """
        A function which draws the current state on the screen and updates the screen. should be called by
        the main loop after every desired object was updated.
        """
        if self._drawing_method == DrawingMethod.FULL:
            self._draw_screen_full()
        elif self._drawing_method == DrawingMethod.SKIP_OBSTACLES:
            if self.iteration_counter % self.full_refresh_rate == 0:
                self._draw_screen_full()
            else:
                self._draw_screen_no_obstacles()
        elif self._drawing_method == DrawingMethod.BACKGROUND_SNAPSHOT:
            self._draw_screen_snapshot()

        # unmark this to display the sensors

        for direction in self.agent.sensors:
            for sensor in self.agent.sensors[direction]:
                start, stop = sensor._create_sensor_line()
                pygame.draw.line(self.window, (255, 255, 255), start, stop)
                min_dot, _ = sensor.detect(self.obstacles_group)
                pygame.draw.circle(self.window, (15, 245, 233), min_dot, 7)

        pygame.display.update()
        self.iteration_counter = (self.iteration_counter + 1) % sys.maxsize

    def get_state(self):
        return self.feature_extractor.get_state(self.parking_lot)

    def do_step(self, movement, steering, time):
        """
        updates the movement of the car
        :param time: the time interval which the car should move
        :param movement: indicates the forward/backward movement of the car
        :param steering: indicates to which side the car should steer
        :return: A dictionary containing data about the results of the current movement:
                    COLLISION: True/False which indicates if the agent collided with the border or with
                               another object.
                    AGENT_IN_UNOCCUPIED: True/False which indicates if there is a free parking slot which
                                         the agent is completely inside it.
                    UNOCCUPIED_PERCENTAGE: A dictionary holding each free parking cell which the agent has
                                           some part of it inside them. the value of each cell is a
                                           percentage between 0 and 1, indicating how much of the agent is
                                           inside that cell.


        """
        self.agent.update(time, movement, steering)
        results = {Results.COLLISION: self.is_collision(),
                   Results.PERCENTAGE_IN_TARGET: self.percentage_in_target_cell()}
        reward = self.reward_analyzer.analyze(self.parking_lot, results)
        return reward, results[Results.COLLISION]

    def is_collision(self):
        """
        :return: True iff the agent collides with any object which should interrupt the agent (other cars,
        walls, sidewalks, etc.), or the agent goes out of bounds.
        """
        agent_rect = self.agent.rect

        # Testing if the agent is in the screen
        window_rect = self.window.get_rect(topleft=(0, 0))
        if not window_rect.contains(agent_rect):
            return True

        # Testing if the agent's rectangle collides with another obstacle's rectangle
        if pygame.sprite.spritecollide(self.agent, self.obstacles_group, False):
            # if there is a collision between the rectangles of the sprites, we will check the masks for
            # better precision (masks are always inside the object's rectangle, so if the rectangles don't
            # collide the masks won't collide either)
            if pygame.sprite.spritecollide(self.agent, self.obstacles_group, False,
                                           pygame.sprite.collide_mask):
                return True
        return False

    def percentage_in_target_cell(self) -> float:
        """
        :return: True iff the agent is fully inside the target parking cell.
        """
        if self.agent.rect.colliderect(self.parking_lot.target_park.rect):
            return mask_subset_percentage(self.parking_lot.target_park, self.agent)
        return 0.0

    def get_agent_percentage_in_unoccupied_cells(self) -> Dict[ParkingCell, float]:
        """
        :return: A dictionary holding each free parking cell which the agent has some part of it inside
        them. the value of each cell is a percentage between 0 and 1, indicating how much of the agent is
        inside that cell.
        """
        collisions = dict()
        for cell in self.parking_lot.free_parking_cells:
            if self.agent.rect.colliderect(cell.rect):
                collisions[cell] = float(mask_subset_percentage(cell, self.agent))
        return collisions

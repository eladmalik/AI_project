import sys
from enum import Enum
from typing import Dict, Union, Callable

import pygame

from simulation.car import Car
from utils.calculations import *
from utils.feature_extractor import FeatureExtractor
from simulation.parking_cell import ParkingCell
from utils.general_utils import mask_subset_percentage
from simulation.parking_lot import ParkingLot
from utils.reward_analyzer import RewardAnalyzer
from utils.enums import Results

from assets.assets_images import ICON_IMG

from utils.lot_generator import LotGenerator

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLOOR = (77, 76, 75)

MAX_SCREEN_SIZE = 1200


class DrawingMethod(Enum):
    FULL = 0
    SKIP_OBSTACLES = 1
    BACKGROUND_SNAPSHOT = 2


class Simulator:
    """
    This class is the driver of the simulator. it initializes the pygame framework, takes inputs from the
    agent/player, outputs their outcome and offers the option to draw them to the screen.
    """

    def __init__(self, lot_generator: LotGenerator, reward_analyzer: Callable[[], RewardAnalyzer],
                 feature_extractor: Callable[[], FeatureExtractor],
                 max_iteration_time_sec: int = 2000,
                 draw_screen: bool = True,
                 resize_screen: bool = True,
                 background_image: Union[pygame.Surface, None] = None,
                 drawing_method=False,
                 full_refresh_rate: int = 30):
        """
        The constructor of the simulator
        :param lot: The parking lot

        :param reward_analyzer: the analyzer responsible for calculating the reward for the current state

        :param feature_extractor: the module responsible for extracting numerical features which describes
        the current state of the simulation

        :param max_iteration_time_sec: the maximum virtual time the simulation should run. virtual time is
        the sum of time differences which was passed to the "do_step" function. typically, when the time
        difference is constant, this implies that: max_steps = max_iteration_time_sec / time_difference

        :param draw_screen: True iff the GUI should be shown on the screen while training/testing. setting
        this to False will improve performance

        :param resize_screen: True iff The GUI's size should stay the same for every episode. setting this
        to True will improve performance

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

               ** THE BEST METHOD SO FAR IS THE BACKGROUND SNAPSHOT, SKIP OBSTACLES IS NOT RECOMMENDED **

        :param full_refresh_rate: if drawing method is set to SKIP_OBSTACLES, the screen will be drawn
               completely once every this number of iterations. higher values means more efficient drawing,
               but less nice-looking screen.
        """
        pygame.font.init()
        self.max_simulator_time = max_iteration_time_sec  # (in seconds)
        self.full_refresh_rate: int = full_refresh_rate
        self.lot_generator = lot_generator
        self.draw_screen = draw_screen
        self.resize_screen = resize_screen
        self.reward_analyzer_class: Callable[[], RewardAnalyzer] = reward_analyzer
        self.feature_extractor_class: Callable[[], FeatureExtractor] = feature_extractor
        self._org_background_image = background_image
        self._drawing_method = drawing_method

        self._init_episode()

        if self.draw_screen:
            if self.resize_screen:
                self.window: pygame.Surface = pygame.display.set_mode((self.width, self.height))
            else:
                self.window: pygame.Surface = pygame.display.set_mode((MAX_SCREEN_SIZE, MAX_SCREEN_SIZE))
            pygame.display.set_icon(ICON_IMG)
            pygame.display.set_caption("Car Parking Simulator")
            self._draw_screen_full()
            pygame.display.update()
        else:
            self.window = pygame.Surface((self.width, self.height))

    def _init_episode(self):
        """
        Initializes the parameters which are relevant to the epochs which is now being initialized
        """
        self.iteration_counter: int = 0
        self.total_time = 0
        self.frame = 0
        self.parking_lot: ParkingLot = self.lot_generator()
        self.width: float = self.parking_lot.width
        self.height: float = self.parking_lot.height
        self.agent: Car = self.parking_lot.car_agent
        self.agent_group: pygame.sprite.Group = pygame.sprite.Group(self.agent)
        self.stationary_cars_group: pygame.sprite.Group = pygame.sprite.Group(
            self.parking_lot.stationary_cars)
        self.parking_cells_group: pygame.sprite.Group = pygame.sprite.Group(self.parking_lot.parking_cells)
        self.obstacles_group: pygame.sprite.Group = pygame.sprite.Group(self.parking_lot.all_obstacles)
        self.borders_group: pygame.sprite.Group = pygame.sprite.Group(self.parking_lot.borders)
        self.reward_analyzer = self.reward_analyzer_class()
        self.feature_extractor = self.feature_extractor_class()
        self.background_img = None
        if self._org_background_image is not None:
            self.background_img = pygame.transform.scale(self._org_background_image,
                                                         (self.parking_lot.width, self.parking_lot.height))
        self._lot_rect = pygame.Rect(0, 0, self.width, self.height)
        self.bg_snapshot = self._create_background_snapshot()
        self.overlays = dict()

    def reset(self):
        """
        re-draws the screen at the beginning of a new simulator iteration
        """
        self._init_episode()

        if self.draw_screen:
            if self.resize_screen:
                if self.width != self.window.get_width() or self.height != self.window.get_height():
                    pygame.display.quit()
                    self.window: pygame.Surface = pygame.display.set_mode((self.width, self.height))
                    pygame.display.set_icon(ICON_IMG)
                    pygame.display.set_caption("Car Parking Simulator")
                self._draw_screen_full()
                pygame.display.update()
            else:
                self.window.fill(BLACK)
                self._draw_screen_full()
                pygame.display.update()
        else:
            self.window = pygame.Surface((self.width, self.height))
        return self.get_state()

    def _create_background_snapshot(self):
        """
        Creates a surface which contains the background and every sprite in the simulator, execpt for the
        agent.
        :return the surface
        """
        snapshot = pygame.Surface((MAX_SCREEN_SIZE, MAX_SCREEN_SIZE))
        if self.background_img is not None:
            snapshot.blit(self.background_img, (0, 0))
        else:
            pygame.draw.rect(snapshot, FLOOR, self._lot_rect)
        self.parking_cells_group.draw(snapshot)
        self.obstacles_group.draw(snapshot)
        return snapshot

    def _draw_screen_full(self, text_dict=None):
        """
        A function which draws all the objects on the screen in the correct order
        """
        # deleting the whole screen and filling it with the floor's color or the background image
        if self.background_img is not None:
            self.window.blit(self.background_img, (0, 0))
        else:
            pygame.draw.rect(self.window, FLOOR, self._lot_rect)
        # drawing the parking cells
        self.parking_cells_group.draw(self.window)
        # drawing the obstacles
        self.obstacles_group.draw(self.window)
        # drawing the agent
        self.agent_group.draw(self.window)

        self.update_info_text_surface(text_dict, font_size=16, text_color=WHITE,
                                      bg_color=BLACK,
                                      location=(MAX_SCREEN_SIZE - 400, 0))
        for surface, rect in self.overlays.values():
            if surface is not None:
                self.window.blit(surface, rect)

    def _draw_screen_no_obstacles(self):
        """
        A function which draws all the objects on the screen in the correct order, but skips the obstacles
        and redraws the background only around the agent
        """
        # deleting the previous car
        if self.background_img is not None:
            sub_rect = self.agent.prev_rect.clip(self._lot_rect)
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

    def _draw_screen_snapshot(self, text_dict):
        """
        A function which redraws the screen only around the agent, using the snapshot image.
        """
        # removing the previous drawing of the agent
        sub_rect = self.agent.prev_rect.clip(self.window.get_rect(topleft=(0, 0)))
        if sub_rect.size != (0, 0):
            subsurface = self.bg_snapshot.subsurface(sub_rect)
            self.window.blit(subsurface, sub_rect.topleft)

        # removing previous overlays
        for surface, rect in self.overlays.values():
            if surface is not None:
                sub_rect = rect.clip(self.window.get_rect(topleft=(0, 0)))
                subsurface = self.bg_snapshot.subsurface(sub_rect)
                self.window.blit(subsurface, sub_rect.topleft)

        # drawing the agent in its new position
        self.agent_group.draw(self.window)

        self.update_info_text_surface(text_dict, font_size=16, text_color=WHITE,
                                      bg_color=BLACK,
                                      location=(MAX_SCREEN_SIZE - 400, 0))

        # drawing new overlays
        for surface, rect in self.overlays.values():
            if surface is not None:
                self.window.blit(surface, rect)

    def update_screen(self, info_text=None):
        """
        A function which draws the current state on the screen and updates the screen. should be called by
        the main loop after every desired object was updated.
        """
        if self._drawing_method == DrawingMethod.FULL:
            self._draw_screen_full(info_text)
        elif self._drawing_method == DrawingMethod.SKIP_OBSTACLES:
            if self.iteration_counter % self.full_refresh_rate == 0:
                self._draw_screen_full(info_text)
            else:
                self._draw_screen_no_obstacles()
        elif self._drawing_method == DrawingMethod.BACKGROUND_SNAPSHOT:
            self._draw_screen_snapshot(info_text)
        pygame.display.update()
        self.iteration_counter = (self.iteration_counter + 1) % sys.maxsize

    def update_info_text_surface(self, info_dict,
                                 font_size=12,
                                 font_style='freesansbold.ttf',
                                 line_spacing=6,
                                 text_color=WHITE,
                                 bg_color=None,
                                 location=(0, 0)):
        """
        this function is used to write text to the GUI
        """
        max_width = 0
        texts = []
        max_height = 0
        surface = None
        rect = None
        if info_dict is not None:
            font = pygame.font.Font(font_style, font_size)
            for key in info_dict:
                text = font.render(f"{key}: {info_dict[key]}", True, text_color)
                text_rect = text.get_rect(topleft=(0, max_height))
                texts.append((text, text_rect))
                if max_width < text_rect.width:
                    max_width = text_rect.width
                max_height += text_rect.height + line_spacing
            surface = pygame.Surface((max_width, max_height))
            if bg_color is not None:
                surface.fill(bg_color)
            else:
                surface.set_colorkey((0, 0, 0))
            for text, text_rect in texts:
                surface.blit(text, text_rect)
            rect = surface.get_rect(topleft=location)
        self.overlays["info_text"] = (surface, rect)

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
        self.total_time += time
        self.frame += 1
        results = {Results.COLLISION: self.is_collision(),
                   Results.PERCENTAGE_IN_TARGET: self.percentage_in_target_cell(),
                   Results.FRAME: self.frame,
                   Results.SIMULATION_TIMEOUT: self.total_time >= self.max_simulator_time,
                   Results.IN_BOUNDS: self._lot_rect.contains(self.agent.rect),
                   Results.DISTANCE_TO_TARGET: get_distance_to_target(self.agent,
                                                                      self.parking_lot.target_park),
                   Results.ANGLE_TO_TARGET: get_angle_to_target(self.agent, self.parking_lot.target_park)}
        results[Results.SUCCESS] = self.reward_analyzer.is_success(self.parking_lot, results)
        reward, done = self.reward_analyzer.analyze(self.parking_lot, results)
        if results[Results.SIMULATION_TIMEOUT]:
            done = True
        new_state = self.get_state()
        return new_state, reward, done, results

    def peek_step(self, movement, steering, time):
        """
        "peeks" the next step that the agent will do, and doesn't actually do it
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
        # save originals
        org_acceleration = self.agent.acceleration
        org_velocity = self.agent.velocity.copy()
        org_steering = self.agent.steering
        org_location = self.agent.location.copy()
        org_rotation = self.agent.rotation

        self.agent.update(time, movement, steering)
        results = {Results.COLLISION: self.is_collision(),
                   Results.PERCENTAGE_IN_TARGET: self.percentage_in_target_cell(),
                   Results.FRAME: self.frame,
                   Results.SIMULATION_TIMEOUT: self.total_time >= self.max_simulator_time,
                   Results.IN_BOUNDS: self._lot_rect.contains(self.agent.rect),
                   Results.DISTANCE_TO_TARGET: get_distance_to_target(self.agent,
                                                                      self.parking_lot.target_park),
                   Results.ANGLE_TO_TARGET: get_angle_to_target(self.agent, self.parking_lot.target_park)}
        results[Results.SUCCESS] = self.reward_analyzer.is_success(self.parking_lot, results)
        reward, done = self.reward_analyzer.analyze(self.parking_lot, results)
        if results[Results.SIMULATION_TIMEOUT]:
            done = True
        new_state = self.get_state()

        # revert original
        self.agent.acceleration = org_acceleration
        self.agent.velocity = org_velocity
        self.agent.steering = org_steering
        self.agent.location = org_location
        self.agent.rotation = org_rotation
        self.agent.update_location(org_location, org_rotation)

        return new_state, reward, done, results

    def is_collision(self):
        """
        :return: True iff the agent collides with any object which should interrupt the agent (other cars,
        walls, sidewalks, etc.), or the agent goes out of bounds.
        """
        agent_rect = self.agent.rect

        # Testing if the agent is in the screen
        window_rect = self._lot_rect
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

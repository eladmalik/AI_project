import os.path
import sys
from enum import Enum
from typing import Dict, Union

import pygame

from parking_cell import ParkingCell
from utils import mask_subset_percentage
from parking_lot import ParkingLot

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLOOR = (77, 76, 75)

PATH_AGENT_IMG = os.path.join("assets", "green-car-top-view.png")
PATH_PARKING_IMG = os.path.join("assets", "parking_left.png")
PATH_PARKING_SIDEWALK_IMG = os.path.join("assets", "parking_sidewalk_down.png")
PATH_CAR_IMG = os.path.join("assets", "orange-car-top-view.png")
PATH_ICON_IMG = os.path.join("assets", "icon.png")
PATH_FLOOR_IMG = os.path.join("assets", "floor.png")


class Results(Enum):
    """
    keys of values returns by the move function of the simulator. use them in order to gather information
    on the current state of the simulation.
    """
    COLLISION = 1
    AGENT_IN_UNOCCUPIED = 2
    UNOCCUPIED_PERCENTAGE = 3


class DrawingMethod(Enum):
    FULL = 0
    SKIP_OBSTACLES = 1
    BACKGROUND_SNAPSHOT = 2


class Simulator:
    """
    This class is the driver of the simulator. it initializes the pygame framework, takes inputs from the
    agent/player, outputs their outcome and offers the option to draw them to the screen.
    """

    def __init__(self, lot: ParkingLot, background_image: Union[str, None] = None,
                 drawing_method=False,
                 full_refresh_rate: int = 30):
        """
        The constructor of the simulator
        :param lot: The parking lot

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
        self.iteration_counter = 0
        self.full_refresh_rate = full_refresh_rate
        self.parking_lot = lot
        self.width = self.parking_lot.width
        self.height = self.parking_lot.height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.agent = self.parking_lot.car_agent
        self.agent_group = pygame.sprite.Group(self.agent)
        self.stationary_cars_group = pygame.sprite.Group(self.parking_lot.stationary_cars)
        self.parking_cells_group = pygame.sprite.Group(self.parking_lot.parking_cells)
        self.obstacles_group = pygame.sprite.Group(self.parking_lot.stationary_cars,
                                                   self.parking_lot.obstacles)

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
        pygame.display.update()
        self.iteration_counter = (self.iteration_counter + 1) % sys.maxsize

    def move_agent(self, movement, steering, time):
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
        return {Results.COLLISION: self.is_collision(),
                Results.AGENT_IN_UNOCCUPIED: self.agent_in_unoccupied_cell(),
                Results.UNOCCUPIED_PERCENTAGE: self.get_agent_percentage_in_unoccupied_cells()}

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

    def agent_in_unoccupied_cell(self) -> bool:
        """
        :return: True iff the agent is fully inside a free parking cell.
        """
        for cell in self.parking_lot.free_parking_cells:
            if self.agent.rect.colliderect(cell.rect):
                collision_percentage = mask_subset_percentage(cell, self.agent)
                if collision_percentage >= 1:
                    return True
        return False

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

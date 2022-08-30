from typing import Tuple, Iterable
import pygame

from simulation.CarSimSprite import CarSimSprite
from utils.enums import SensorDirection


class ProximitySensor:
    def __init__(self, car, direction: SensorDirection, angle: float, max_distance: float, use_mask=True):
        self.car = car
        self.direction: SensorDirection = direction
        self.angle = angle
        self.max_distance = max_distance
        self.use_mask = use_mask

    def _create_sensor_line(self) -> Tuple[pygame.Vector2, pygame.Vector2]:
        """
        this function creates the sensor's beam
        """
        start_pos = self.car.get_sensor_start_point()[self.direction]
        stop_pos = start_pos + self.max_distance * self.car.get_direction_vectors(self.angle)[self.direction]

        return start_pos, stop_pos

    def detect(self, obstacles: Iterable[CarSimSprite]) -> Tuple[pygame.Vector2, float]:
        """
        this function makes the sensor recognize the closest object that collides
        with its beam
        """
        start_pos, end_pos = self._create_sensor_line()
        min_pos = end_pos
        for obstacle in obstacles:
            points = obstacle.rect.clipline(start_pos.x, start_pos.y, end_pos.x, end_pos.y)
            if len(points) == 0:
                continue
            point1, point2 = points
            temp_min = point1
            if self.use_mask:
                temp_surface = pygame.Surface(obstacle.rect.size)
                temp_surface.set_colorkey((0, 0, 0))
                pygame.draw.line(temp_surface, (255, 255, 255),
                                 point1 - pygame.Vector2(obstacle.rect.topleft),
                                 point2 - pygame.Vector2(obstacle.rect.topleft))

                line_mask = pygame.mask.from_surface(temp_surface)
                overlap = obstacle.mask.overlap_mask(line_mask, (0, 0)).get_bounding_rects()
                if len(overlap) == 0:
                    continue
                overlap = overlap[0]
                absolute_rect = pygame.Rect(obstacle.rect.topleft[0] + overlap.x, obstacle.rect.topleft[
                    1] + overlap.y, overlap.width, overlap.height)
                points = absolute_rect.clipline(start_pos.x, start_pos.y, end_pos.x, end_pos.y)
                if len(points) == 0:
                    continue
                point1, point2 = points
                temp_min = point1

            if start_pos.distance_squared_to(temp_min) < start_pos.distance_squared_to(min_pos):
                min_pos = temp_min
        return min_pos, start_pos.distance_to(min_pos)

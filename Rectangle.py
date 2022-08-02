from typing import Tuple


class Rectangle:
    def __init__(self, width: float, height: float, top_left_pos: Tuple[float, float] = None,
                 center_pos: Tuple[float, float] = None, rotation: float = 0):
        assert 0 <= rotation < 360, "Rotation must be in range of [0,360) (in degrees)"
        assert bool(top_left_pos is None) != bool(center_pos is None), "You must supply exactly one of the " \
                                                                       "following: top_left_pos, center_pos"
        self.width = width
        self.height = height
        self.rotation = rotation
        self.top_left_pos = None
        self.center_pos = None

    def calculate_center_from_topleft(self):
        pass
    def calculate_topleft_from_center(self):
        pass

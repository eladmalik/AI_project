import os.path

from assets_paths import PATH_SIDEWALK_IMG
from CarSimSprite import CarSimSprite


class Sidewalk(CarSimSprite):
    DEFAULT_IMAGE_PATH = PATH_SIDEWALK_IMG

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 img_path: str = DEFAULT_IMAGE_PATH, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, img_path, topleft)

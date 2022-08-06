import os.path

from CarSimSprite import CarSimSprite


class Sidewalk(CarSimSprite):
    DEFAULT_IMAGE_PATH = os.path.join("assets", "sidewalk2.png")

    def __init__(self, x: float, y: float, width: float, height: float, rotation: float,
                 img_path: str = DEFAULT_IMAGE_PATH, topleft: bool = False):
        super().__init__(x, y, width, height, rotation, img_path, topleft)

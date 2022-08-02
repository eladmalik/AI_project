class ParkingCell:
    def __init__(self, x, y, width, height, rotation, car=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rotation = rotation
        self.car = car

    def is_occupied(self):
        return self.car is not None

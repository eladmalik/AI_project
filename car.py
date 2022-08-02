class Car:
    def __init__(self, init_x, init_y, width, height, rotation):
        self.x = init_x
        self.y = init_y
        self.width = width
        self.height = height
        self.rotation = rotation


class AgentCar(Car):
    def __init__(self, init_x, init_y, width, height, rotation):
        super().__init__(init_x, init_y, width, height, rotation)




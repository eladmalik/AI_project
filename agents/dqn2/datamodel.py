import random
from collections import namedtuple, deque
from typing import Tuple, List


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ActionType = Tuple[int, int, int, int, int, int, int, int, int, int, int, int]

Actions: List[ActionType] = [
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),   # (0) - (Movement.NEUTRAL, Steering.NEUTRAL)
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),   # (1) - (Movement.NEUTRAL, Steering.LEFT)
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),   # (2) - (Movement.NEUTRAL, Steering.RIGHT)
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),   # (3) - (Movement.FORWARD, Steering.NEUTRAL)
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),   # (4) - (Movement.FORWARD, Steering.LEFT)
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),   # (5) - (Movement.FORWARD, Steering.RIGHT)
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),   # (6) - (Movement.BACKWARD, Steering.NEUTRAL)
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),   # (7) - (Movement.BACKWARD, Steering.LEFT)
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),   # (8) - (Movement.BACKWARD, Steering.RIGHT)
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),   # (9) - (Movement.BRAKE, Steering.NEUTRAL)
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),   # (10) - (Movement.BRAKE, Steering.LEFT)
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)    # (11) - (Movement.BRAKE, Steering.RIGHT)
]


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
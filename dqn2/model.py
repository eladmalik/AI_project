from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
import os

import utils


class DQNAgent(ABC, nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def save(self, folder='./model/', file_name=f'dqn.pth'):
        file_name = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_name)


# class DQNAgent1(DQNAgent):
#     def __init__(self, input_size: int, output_size: int, hidden_size1):
#         super().__init__(input_size, output_size)
#         hidden_size1 = int(hidden_size1)

#         self.time_created = utils.get_time()
#         self.network = nn.Sequential(nn.Linear(input_size, hidden_size1),
#                                      nn.ReLU(),
#                                      nn.Linear(hidden_size1, output_size))

#     def forward(self, x):
#         return self.network(x)



class DQNAgent2(DQNAgent):
    def __init__(self, input_size, output_size):
        super().__init__(int(input_size), int(output_size))
        input_size = int(input_size)
        hidden_size1 = int(128)
        hidden_size2 = int(256)
        hidden_size3 = int(512)
        hidden_size4 = int(128)
        output_size = int(output_size)

        self.time_created = utils.get_time()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size1),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size1, hidden_size2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size2, hidden_size3),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size3, hidden_size4),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size4, output_size))

    def forward(self, x):
        return self.network(x)
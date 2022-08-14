import torch
from torch import nn
import torch.nn.functional as F
import os

import utils


class DQNAgent1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1):
        super().__init__()
        input_size = int(input_size)
        hidden_size1 = int(hidden_size1)
        output_size = int(output_size)

        self.time_created = utils.get_time()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, folder='./model/', file_name=f'dqn.pth'):
        file_name = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_name)


class DQNAgent2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, hidden_size3):
        super().__init__()
        input_size = int(input_size)
        hidden_size1 = int(hidden_size1)
        hidden_size2 = int(hidden_size2)
        hidden_size3 = int(hidden_size3)
        output_size = int(output_size)

        self.time_created = utils.get_time()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, folder='./model/', file_name=f'dqn.pth'):
        file_name = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_name)

import torch
from torch import nn
import torch.nn.functional as F
import os

import utils


class DQNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.time_created = utils.get_time()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name=f'dqn.pth'):
        file_name = file_name[:file_name.rfind(".")] + f"_{self.time_created}" + file_name[
                                                                                 file_name.rfind("."):]
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DQNAgent2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
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

    def save(self, file_name=f'dqn.pth'):
        file_name = file_name[:file_name.rfind(".")] + f"_{self.time_created}" + file_name[
                                                                                 file_name.rfind("."):]
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

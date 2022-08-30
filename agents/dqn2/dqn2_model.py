import pickle

from torch import nn
import os

STRUCTURE_NAME = "structure.pickle"
PTH_NAME = "agent.pth"


class DQN_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=128, hidden_size2=128, hidden_size3=128,
                 hidden_size4=128,
                 hidden_size5=128):
        super().__init__()
        self.input_size = input_size
        self.actions_num = output_size
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size1),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size1, hidden_size2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size2, hidden_size3),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size3, hidden_size4),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size4, hidden_size5),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size5, output_size))

    def forward(self, x):
        return self.network(x)


class FlatDQN_Model(nn.Module):
    def __init__(self, input_size, output_size, save_folder="tmp"):
        super().__init__()
        self.save_folder = save_folder
        self.input_size = input_size
        self.actions_num = output_size
        self.network = nn.Linear(input_size, output_size)
        with open(os.path.join(save_folder, STRUCTURE_NAME), "wb") as file:
            pickle.dump(self.network, file)

    def forward(self, x):
        return self.network(x)

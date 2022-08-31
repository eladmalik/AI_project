import os
import pickle

import torch
from torch import nn

DEFAULT_NAME = 'model.pth'
MODEL_STRUCT = 'seq.pickle'

class GeneticModel(nn.Module):
    def __init__(self, input_dims, n_actions,
                 fc1_dims=128, fc2_dims=128, fc3_dims=128, save_folder=os.path.join("tmp", "gen")):
        super(GeneticModel, self).__init__()

        self.checkpoint_file = os.path.join(save_folder, DEFAULT_NAME)
        self.checkpoint_dir = save_folder
        self.model = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        with open(os.path.join(save_folder, MODEL_STRUCT), "wb") as file:
            pickle.dump(self.model, file)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.model(state)

        return dist

    def change_checkpoint_dir(self, new_dir):
        self.checkpoint_dir = new_dir
        self.checkpoint_file = os.path.join(new_dir, DEFAULT_NAME)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        with open(os.path.join(self.checkpoint_dir, MODEL_STRUCT), "rb") as file:
            self.model = pickle.load(file)
        self.load_state_dict(torch.load(self.checkpoint_file))

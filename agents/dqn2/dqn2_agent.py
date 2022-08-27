import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from typing import Type
import math
import random
import os

from agents.dqn2.datamodel import ReplayMemory, Transition
from agents.dqn2.dqn2_model import DQN_Model
from utils.general_utils import action_mapping

from utils.feature_extractor import FeatureExtractor

#
# based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#


DQNNetwork = DQN_Model

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from gym action space
n_actions = len(action_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PTH_NAME = "policy_net.pth"
MODEL_STRUCT = "structure.pth"


class DQNReinforcmentAgent:
    def __init__(self, DQN, Extractor: Type[FeatureExtractor], max_mem=10000,
                 batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.3,
                 eps_decay=5000000, target_update=1, save_folder="tmp") -> None:

        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.target_update: int = target_update
        self.memory = ReplayMemory(max_mem)
        self.save_folder = save_folder

        self.policy_net = DQN(Extractor.input_num, n_actions).to(device)
        self.target_net = DQN(Extractor.input_num, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        with open(os.path.join(self.save_folder, MODEL_STRUCT), "wb") as file:
            pickle.dump(self.policy_net, file)

        self.steps_done = 0

    def get_action(self, state):
        eps_threshold = max(self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done /
                                                                                      self.eps_decay), 0.4)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).argmax(), False
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), True

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, iteration: int = None):
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        torch.save(self.policy_net.state_dict(), os.path.join(self.save_folder, name))

    def load(self, iteration: int = None):
        if os.path.exists(os.path.join(self.save_folder, MODEL_STRUCT)):
            with open(os.path.join(self.save_folder, MODEL_STRUCT), "rb") as file:
                self.policy_net = pickle.load(file)
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        self.policy_net.load_state_dict(torch.load(os.path.join(self.save_folder, name)))
        self.target_net.load_state_dict(self.policy_net.state_dict())

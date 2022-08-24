
import torch
import torch.optim as optim
import torch.nn as nn

from typing import Type
import math
import random
import utils
import os
import numpy as np

from dqn2.datamodel import Actions, ReplayMemory, Transition
from dqn2.model import DQNAgent, DQNAgent2

from training_utils.feature_extractor import FeatureExtractor

#
# based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.3
EPS_DECAY = 5000000
TARGET_UPDATE = 1

DQNNetwork = DQNAgent2
memory = ReplayMemory(10000)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from gym action space
n_actions = len(Actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNReinforcmentAgent:
    def __init__(self, DQN: Type[DQNAgent], Extractor: Type[FeatureExtractor], max_mem=10000) -> None:
        self.policy_net = DQN(Extractor.input_num, n_actions).to(device)
        self.target_net = DQN(Extractor.input_num, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(max_mem)

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
        self.steps_done = 0
        self.episode_scores = []

    def get_action(self, state):
        eps_threshold = max(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY), 0.4)
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
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
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
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, episode_idx: int):
        filename = f'{self.policy_net.__class__.__name__}_{utils.get_time()}_eps{episode_idx}.pth'
        folder = f'./model/{filename}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.policy_net.save(folder, filename)

    def load(self, filename: str):
        self.policy_net.load_state_dict(torch.load(filename, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
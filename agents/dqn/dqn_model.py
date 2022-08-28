import pickle
import random
from abc import ABC
from collections import deque

import torch
from torch import nn, optim as optim
import torch.nn.functional as F
import os

import utils
from simulation.simulator import Simulator

STRUCTURE_NAME = "structure.pickle"
PTH_NAME = "agent.pth"


class DQN_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=128, hidden_size2=128, hidden_size3=128,
                 hidden_size4=128,
                 hidden_size5=128, save_folder="tmp"):
        super().__init__()
        self.save_folder = save_folder
        self.input_size = input_size
        self.actions_num = output_size
        # self.network = nn.Sequential(nn.Linear(input_size, hidden_size1),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_size1, hidden_size2),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_size2, hidden_size3),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_size3, hidden_size4),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_size4, hidden_size5),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_size5, output_size))
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size1),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size1, hidden_size2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size2, hidden_size3),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size3, output_size))
        with open(os.path.join(save_folder, STRUCTURE_NAME), "wb") as file:
            pickle.dump(self.network, file)

    def forward(self, x):
        return self.network(x)

    def save(self, iteration=None):
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        torch.save(self.state_dict(), os.path.join(self.save_folder, name))

    def load(self, iteration=None):
        if os.path.exists(os.path.join(self.save_folder, STRUCTURE_NAME)):
            with open(os.path.join(self.save_folder, STRUCTURE_NAME), "rb") as file:
                self.network = pickle.load(file)
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        self.load_state_dict(torch.load(os.path.join(self.save_folder, name)))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Agent:

    def __init__(self, simulator: Simulator, model,
                 randomness_rate=0.25,
                 learning_rate=0.001,
                 gamma=0.9,
                 max_epsilon=1000,
                 batch_size=1000,
                 max_memory=100000):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.randomness_rate = randomness_rate
        self.gamma = gamma  # discount rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.simulator = simulator
        self.max_epsilon = max_epsilon
        self.memory = deque(maxlen=self.max_memory)  # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.max_epsilon - self.n_games
        if self.randomness_rate > 0 and \
                random.randint(0, int(self.max_epsilon / self.randomness_rate)) < self.epsilon:
            move = random.randint(0, self.model.actions_num - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move

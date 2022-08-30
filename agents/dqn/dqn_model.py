import pickle
import random
from abc import ABC, abstractmethod
from collections import deque

import torch
from torch import nn, optim as optim
import os
from simulation.simulator import Simulator

STRUCTURE_NAME = "structure.pickle"
PTH_NAME = "agent.pth"


class DQNModel(ABC, nn.Module):
    """
    The basic structure of the DQN's neural network
    """

    @abstractmethod
    def forward(self, x):
        ...


class DQN_Model(DQNModel):
    def __init__(self, input_size, output_size, hidden_size1=128, hidden_size2=128, hidden_size3=128,
                 hidden_size4=128,
                 hidden_size5=128,
                 save_folder="tmp"):
        super().__init__()
        self.input_size = input_size
        self.actions_num = output_size
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


class FlatDQN_Model(DQNModel):
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


class QTrainer:
    """
    A class which is responsible for training the DQN
    """
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
    """
    The DQN Agent
    """

    def __init__(self, simulator: Simulator, model,
                 randomness_rate=0.25,
                 learning_rate=0.001,
                 gamma=0.9,
                 max_epsilon=1000,
                 batch_size=1000,
                 max_memory=100000,
                 save_folder="tmp",
                 is_eval=False):
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
        self.save_folder = save_folder
        self.is_eval = is_eval

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the results of an action to the memory, to be learned later
        """
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        """
        Used after an epoch, trains the agent on a large batch of data at once
        """
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the agent on a single action
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Returns an action according to the given state. might return a random value in order to explore the
        state space
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.max_epsilon - self.n_games
        if self.randomness_rate > 0 and \
                random.randint(0, int(self.max_epsilon / self.randomness_rate)) < self.epsilon:
            move = random.randint(0, self.model.actions_num - 1)
        else:
            if self.is_eval:
                with torch.no_grad():
                    state0 = torch.tensor(state, dtype=torch.float)
                    prediction = self.model(state0)
                    move = torch.argmax(prediction).item()
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()

        return move

    def save(self, iteration=None):
        """
        Saves the current state of the agent
        """
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.save_folder, name))

    def load(self, iteration=None):
        """
        Loads a state of the agent
        """
        if os.path.exists(os.path.join(self.save_folder, STRUCTURE_NAME)):
            with open(os.path.join(self.save_folder, STRUCTURE_NAME), "rb") as file:
                self.network = pickle.load(file)
        name = PTH_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        self.model.load_state_dict(torch.load(os.path.join(self.save_folder, name)))

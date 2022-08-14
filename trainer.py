import os
from collections import deque
import random

import pygame.event
import torch
from torch import nn
import torch.optim as optim
import configparser

import lot_generator
import utils
from assets_paths import PATH_FLOOR_IMG
from dqn_model import DQNAgent1, DQNAgent2
from feature_extractor import Extractor
from reward_analyzer import Analyzer, AnalyzerPenaltyOnStanding, AnalyzerStopOnTarget
from simulator import Simulator, DrawingMethod
from car import Movement, Steering

MAX_MEMORY = 100000
BATCH_SIZE = 1000

NUM_OF_ACTIONS = 12

config = configparser.ConfigParser()
config.read("training_settings.ini")
conf_default = config["DEFAULT"]
conf_model_load = config["LoadModel"]

Analyzers = {
    "Analyzer": Analyzer,
    "AnalyzerPenaltyOnStanding": AnalyzerPenaltyOnStanding,
    "AnalyzerStopOnTarget": AnalyzerStopOnTarget
}

Extractors = {
    "Extractor": Extractor
}

Model_Classes = {
    "DQNAgent": DQNAgent1,
    "DQNAgent2": DQNAgent2
}

MOVEMENT_STEERING_TO_ACTION = {
    (Movement.NEUTRAL, Steering.NEUTRAL): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    (Movement.NEUTRAL, Steering.LEFT): [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    (Movement.NEUTRAL, Steering.RIGHT): [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    (Movement.FORWARD, Steering.NEUTRAL): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    (Movement.FORWARD, Steering.LEFT): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    (Movement.FORWARD, Steering.RIGHT): [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    (Movement.BACKWARD, Steering.NEUTRAL): [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    (Movement.BACKWARD, Steering.LEFT): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    (Movement.BACKWARD, Steering.RIGHT): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    (Movement.BRAKE, Steering.NEUTRAL): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    (Movement.BRAKE, Steering.LEFT): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    (Movement.BRAKE, Steering.RIGHT): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

}

ACTION_TO_MOVEMENT_STEERING = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (Movement.NEUTRAL, Steering.NEUTRAL),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (Movement.NEUTRAL, Steering.LEFT),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0): (Movement.NEUTRAL, Steering.RIGHT),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): (Movement.FORWARD, Steering.NEUTRAL),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0): (Movement.FORWARD, Steering.LEFT),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): (Movement.FORWARD, Steering.RIGHT),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): (Movement.BACKWARD, Steering.NEUTRAL),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0): (Movement.BACKWARD, Steering.LEFT),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0): (Movement.BACKWARD, Steering.RIGHT),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): (Movement.BRAKE, Steering.NEUTRAL),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0): (Movement.BRAKE, Steering.LEFT),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1): (Movement.BRAKE, Steering.RIGHT)
}


def load_model():
    loaded_config = configparser.ConfigParser()
    loaded_config.read(os.path.join(conf_model_load["model_folder_path"], "training_settings.ini"))
    config._sections[conf_default["model"]] = loaded_config._sections[conf_default["model"]]

    kwargs = dict(config._sections[conf_default["model"]])
    kwargs["input_size"] = Extractors[conf_default["extractor"]]().input_num
    kwargs["output_size"] = NUM_OF_ACTIONS
    model = Model_Classes[conf_default["model"]](**kwargs)
    model.load_state_dict(torch.load(os.path.join
                                     (conf_model_load["model_folder_path"],
                                      conf_model_load["model_filename"])))
    model.eval()
    return model


def get_agent_output_folder():
    filename = f'{conf_default["model"]}_{utils.get_time()}.pth'
    folder = f'./model/{filename}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder, filename


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


class AgentTrainer:

    def __init__(self, simulator: Simulator, model):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = float(conf_default["gamma"])  # discount rate
        self.learning_rate = float(conf_default["learning_rate"])
        self.simulator = simulator
        self.max_epsilon = int(conf_default["max_epsilon"])
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.max_epsilon - self.n_games  # TODO: change
        final_move = [0] * NUM_OF_ACTIONS
        if random.randint(0, self.max_epsilon * 4) < self.epsilon:
            move = random.randint(0, NUM_OF_ACTIONS - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return tuple(final_move)


def train():
    plot_rewards = []
    plot_distance = []
    plot_mean_distance = []
    time_difference = float(conf_default["time_difference"])
    draw_screen = bool(int(conf_default["draw_screen"]))
    total_score = 0
    record = 0
    iteration_max_reward = 0
    lot = lot_generator.generate_lot()
    sim = Simulator(lot, Analyzers[conf_default["analyzer"]](), Extractors[conf_default["extractor"]](),
                    draw_screen=draw_screen,
                    max_iteration_time_sec=int(conf_default["max_iteration_time_sec"]),
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
                    background_image=PATH_FLOOR_IMG)
    folder, filename = get_agent_output_folder()
    if bool(int(conf_model_load["load"])):
        model = load_model()
    else:
        with open(os.path.join(folder, "training_settings.ini"), "w") as configfile:
            config.write(configfile)
        kwargs = dict(config._sections[conf_default["model"]])
        kwargs["input_size"] = Extractors[conf_default["extractor"]]().input_num
        kwargs["output_size"] = NUM_OF_ACTIONS
        model = Model_Classes[conf_default["model"]](**kwargs)

    agent_trainer = AgentTrainer(sim, model)
    while True:
        # get old state
        state_old = agent_trainer.simulator.get_state()

        # get move
        final_move = agent_trainer.get_action(state_old)

        # perform move and get new state
        final_movement, final_steering = ACTION_TO_MOVEMENT_STEERING[final_move]
        reward, done = agent_trainer.simulator.do_step(final_movement, final_steering, time_difference)
        if draw_screen:
            pygame.event.pump()
            agent_trainer.simulator.update_screen()
        if reward > iteration_max_reward:
            iteration_max_reward = reward
        state_new = agent_trainer.simulator.get_state()

        # train short memory
        agent_trainer.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent_trainer.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            print(f"Total real time: {agent_trainer.simulator.total_time}")
            plot_distance.append(sim.agent.location.distance_to(sim.parking_lot.target_park.location))
            plot_mean_distance.append(sum(plot_distance) / len(plot_distance))

            lot = lot_generator.generate_lot()
            sim = Simulator(lot, Analyzers[conf_default["analyzer"]](),
                            Extractors[conf_default["extractor"]](),
                            draw_screen=draw_screen,
                            max_iteration_time_sec=int(conf_default["max_iteration_time_sec"]),
                            drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
                            background_image=PATH_FLOOR_IMG)
            agent_trainer.simulator = sim
            agent_trainer.n_games += 1
            agent_trainer.train_long_memory()

            if iteration_max_reward > record:
                record = iteration_max_reward
            agent_trainer.model.save(folder, filename)

            print('Game', agent_trainer.n_games, 'Reward', iteration_max_reward, 'Record:', record)

            plot_rewards.append(iteration_max_reward)
            if agent_trainer.n_games % 50 == 0:
                utils.plot(plot_distance, plot_mean_distance)

            iteration_max_reward = 0


if __name__ == '__main__':
    train()
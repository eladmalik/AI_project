import configparser
from math import ceil
from optparse import check_choice
from random import random
from typing import Optional
from tqdm import tqdm
from itertools  import count

import pygame

import torch

from dqn2.dqn import DQNReinforcmentAgent, TARGET_UPDATE
from dqn2.model import DQNAgent2
from dqn2.datamodel import Actions
from sim.simulator import Simulator
from dqn.trainer import Lot_Generators, Analyzers, Extractors, DrawingMethod, ACTION_TO_MOVEMENT_STEERING, get_agent_output_folder
from training_utils.feature_extractor import ExtractorNew

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("./dqn2/training_settings.ini")
conf_default = config["DEFAULT"]

time_difference = float(conf_default["time_difference"])
draw_screen = bool(int(conf_default["draw_screen"]))
draw_rate = int(conf_default["draw_rate"])
resize_screen = bool(int(conf_default["resize_screen"]))
MODEL_SAVE_RATE = int(conf_default["model_save_rate"])

def train(checkpoint: Optional[str] = None):    
    iteration_total_reward = 0

    sim = Simulator(Lot_Generators[conf_default["lot_generation"]],
                    Analyzers[conf_default["analyzer"]],
                    Extractors[conf_default["extractor"]],
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    max_iteration_time_sec=int(conf_default["max_iteration_time_sec"]),
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)

    num_steps = int(ceil(sim.max_simulator_time / time_difference))
    progress_bar = tqdm(range(num_steps))
    progress_bar_iter = progress_bar.__iter__()
    iter_num = 0
    i_episode = 0
    plot_distance = []
    plot_mean_distance = []
    plot_rewards = []

    agent = DQNReinforcmentAgent(DQNAgent2, ExtractorNew)

    if checkpoint is not None:
        print(f"loading model checkpoint {checkpoint}")
        agent.load(checkpoint)

    while True:
        # episodes
        sim.reset()
        i_episode += 1
        state = torch.as_tensor(sim.get_state(), device=device)

        for t in count():
            # Select and perform an action
            action, is_random = agent.get_action(state)
            action_vec = Actions[int(action.item())]
            next_state, reward, done = sim.do_step(*ACTION_TO_MOVEMENT_STEERING[action_vec], time_difference)
            
            action = torch.as_tensor(action, device=device)
            next_state = torch.as_tensor(next_state, device=device)
            reward_torch = torch.tensor([reward], device=device)

            # Store the transition in memory
            agent.remember(state, action, next_state, reward_torch)

            # Move to the next state
            state = next_state

            if is_random and random() < 0.5:
                for _ in range(3):
                    next_state, reward, done = sim.do_step(*ACTION_TO_MOVEMENT_STEERING[action_vec], time_difference)
                    next_state = torch.as_tensor(next_state, device=device)
                    reward_torch = torch.tensor([reward], device=device)
                    agent.remember(state, action, next_state, reward_torch)
                    state = next_state

            # manage logging and screen output
            if draw_screen:
                try:
                    next(progress_bar_iter)   
                except:
                    pass
                if agent.steps_done % 60 == 0:
                    progress_bar.set_description(f"reward: {reward: .2f}")                                                             
                if i_episode % draw_rate == 0:
                    pygame.event.pump()
                    sim.update_screen()
            iteration_total_reward += reward

            # Perform one step of the optimization (on the policy network)
            agent.optimize()
            if done:
                # train long memory, plot result
                print(f"Total real time: {sim.total_time}")
                final_distance = sim.agent.location.distance_to(sim.parking_lot.target_park.location)
                plot_distance.append(final_distance)
                plot_mean_distance.append(sum(plot_distance) / len(plot_distance))

                progress_bar = tqdm(range(num_steps))
                progress_bar_iter = progress_bar.__iter__()

                if i_episode % MODEL_SAVE_RATE == 0:
                    agent.save(i_episode)

                print('Game', i_episode, 'Reward', iteration_total_reward, 'Distance', f"{final_distance:.2f}")

                plot_rewards.append(iteration_total_reward)

                iteration_total_reward = 0
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target()

        if i_episode > 200:
            print("done")
            return
import os

import pygame.event
import torch

from utils.csv_handler import csv_handler
from utils.lot_generator import *
import utils.general_utils
from utils.general_utils import action_mapping, dump_arguments
from assets.assets_images import FLOOR_IMG
from agents.dqn.dqn_model import Agent, DQN_Model
from utils.feature_extractor import *
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod

MAX_MEMORY = 100000
BATCH_SIZE = 1000

AGENT_TYPE = "DQN"

Analyzers = {
    "Analyzer": Analyzer,
    "AnalyzerPenaltyOnStanding": AnalyzerPenaltyOnStanding,
    "AnalyzerStopOnTarget": AnalyzerStopOnTarget,
    "AnalyzerDistanceCritical": AnalyzerDistanceCritical,
    "AnalyzerCollisionReduceNearTarget": AnalyzerCollisionReduceNearTarget,
    "AnalyzerNoCollision": AnalyzerNoCollision,
    "AnalyzerNoCollisionNoDistanceReward": AnalyzerNoCollisionNoDistanceReward,
    "AnalyzerAccumulating": AnalyzerAccumulating,
    "AnalyzerAccumulating2": AnalyzerAccumulating2,
    "AnalyzerAccumulating3": AnalyzerAccumulating3,
    "AnalyzerAccumulating4": AnalyzerAccumulating4,
    "AnalyzerNew": AnalyzerNew,
    "AnalyzerAccumulating4FrontBack": AnalyzerAccumulating4FrontBack
}

Extractors = {
    "Extractor": Extractor,
    "Extractor2": Extractor2,
    "Extractor2NoSensors": Extractor2NoSensors,
    "Extractor3": Extractor3,
    "Extractor4": Extractor4,
    "ExtractorNew": ExtractorNew,
    "Extractor8": Extractor8
}

Lot_Generators = {
    "random": generate_lot,
    "example0": example0,
    "example1": example1,
    "only_target": generate_only_target,
    "example2": example2
}


@dump_arguments(agent_type=AGENT_TYPE)
def main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor8,
         load_model=False,
         load_folder=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=True,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         batch_size=1000,
         max_memory=100000,
         randomness_rate=0.25,
         learning_rate=0.0005,
         gamma=0.98,
         plot_interval=100,
         checkpoint_interval=250,
         save_folder=None):
    assert (not load_model) or (load_model and isinstance(load_folder, str))
    plot_rewards = []
    plot_distance = []
    plot_mean_distance = []
    iteration_total_reward = 0
    sim = Simulator(lot_generator,
                    reward_analyzer,
                    feature_extractor,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    max_iteration_time_sec=max_iteration_time,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    if save_folder is None:
        save_folder = utils.general_utils.get_agent_output_folder(AGENT_TYPE)

    model = DQN_Model(feature_extractor.input_num, len(utils.general_utils.action_mapping), 128, 128, 128,
                      128,
                      128, save_folder)
    agent = Agent(sim, model,
                  learning_rate=learning_rate,
                  gamma=gamma,
                  randomness_rate=randomness_rate,
                  batch_size=batch_size,
                  max_memory=max_memory)
    if load_model:
        model.save_folder = load_folder
        model.load()
        model.save_folder = save_folder

    result_writer = csv_handler(save_folder, ["last reward", "total reward", "distance to target",
                                             "percentage in target", "angle to target", "success",
                                             "collision"])

    while True:
        # get old state
        state_old = agent.simulator.get_state()

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        state_new, reward, done, results = agent.simulator.do_step(*action_mapping[action],
                                                                 time_difference_secs)
        if draw_screen and agent.n_games % draw_rate == 0:
            pygame.event.pump()
            agent.simulator.update_screen()
        iteration_total_reward += reward

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            print(f"Total real time: {agent.simulator.total_time}")
            print('Simulation', agent.n_games, 'Reward', iteration_total_reward, 'Distance',
                  f"{results[Results.DISTANCE_TO_TARGET]:.2f}")
            result_writer.write_row({
                "last reward": reward,
                "total reward": iteration_total_reward,
                "distance to target": results[Results.DISTANCE_TO_TARGET],
                "percentage in target": results[Results.PERCENTAGE_IN_TARGET],
                "angle to target": results[Results.ANGLE_TO_TARGET],
                "success": results[Results.SUCCESS],
                "collision": results[Results.COLLISION]
            })

            sim.reset()
            agent.n_games += 1
            agent.train_long_memory()

            agent.model.save()
            if agent.n_games % checkpoint_interval == 0:
                agent.model.save(custom_name=f"iter_{agent.n_games}.pth")

            plot_rewards.append(iteration_total_reward)
            if agent.n_games % plot_interval == 0:
                utils.general_utils.plot_distances(plot_distance, plot_mean_distance, save_folder)

            iteration_total_reward = 0
            if agent.n_games >= n_simulations:
                break


if __name__ == '__main__':
    main()

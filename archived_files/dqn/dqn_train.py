import os

if __name__ == '__main__':
    os.chdir(os.path.join("../../agents", ".."))
import pygame.event

from utils.csv_handler import csv_handler
from utils.enums import StatsType
from utils.lot_generator import *
import utils.general_utils
from utils.general_utils import action_mapping, dump_arguments
from archived_files.dqn.dqn_model import Agent, DQN_Model
from utils.feature_extractor import *
from utils.plots.make_plots import plot_all_from_lines
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "DQN"


@dump_arguments(agent_type=AGENT_TYPE)
def main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor9,
         load_model=False,
         load_folder=None,
         load_iter=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=False,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         batch_size=1000,
         max_memory=100000,
         randomness_rate=0.33,
         learning_rate=0.01,
         gamma=0.9,
         plot_in_training=True,
         plot_interval=100,
         checkpoint_interval=250,
         save_folder=None):
    assert (not load_model) or (load_model and isinstance(load_folder, str))

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
                      128)
    agent = Agent(sim, model,
                  learning_rate=learning_rate,
                  gamma=gamma,
                  randomness_rate=randomness_rate,
                  batch_size=batch_size,
                  max_memory=max_memory,
                  save_folder=save_folder)
    if load_model:
        agent.save_folder = load_folder
        agent.load(load_iter)
        agent.save_folder = save_folder

    result_writer = csv_handler(save_folder, [StatsType.LAST_REWARD,
                                              StatsType.TOTAL_REWARD,
                                              StatsType.DISTANCE_TO_TARGET,
                                              StatsType.PERCENTAGE_IN_TARGET,
                                              StatsType.ANGLE_TO_TARGET,
                                              StatsType.SUCCESS,
                                              StatsType.COLLISION])

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
            print(f"Total virtual time: {agent.simulator.total_time}")
            print('Simulation', agent.n_games, 'Reward', iteration_total_reward, 'Distance',
                  f"{results[Results.DISTANCE_TO_TARGET]:.2f}")
            result_writer.write_row({
                StatsType.LAST_REWARD: reward,
                StatsType.TOTAL_REWARD: iteration_total_reward,
                StatsType.DISTANCE_TO_TARGET: results[Results.DISTANCE_TO_TARGET],
                StatsType.PERCENTAGE_IN_TARGET: results[Results.PERCENTAGE_IN_TARGET],
                StatsType.ANGLE_TO_TARGET: results[Results.ANGLE_TO_TARGET],
                StatsType.SUCCESS: results[Results.SUCCESS],
                StatsType.COLLISION: results[Results.COLLISION]
            })

            sim.reset()
            agent.n_games += 1
            agent.train_long_memory()

            agent.save()
            if agent.n_games % checkpoint_interval == 0:
                agent.save(iteration=agent.n_games)

            if agent.n_games % plot_interval == 0:
                plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

            iteration_total_reward = 0
            if agent.n_games >= n_simulations:
                break


if __name__ == '__main__':
    main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor9,
         load_model=False,
         load_folder=None,
         load_iter=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=True,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         batch_size=1000,
         max_memory=100000,
         randomness_rate=0.25,
         learning_rate=0.001,
         gamma=0.98,
         plot_in_training=True,
         plot_interval=100,
         checkpoint_interval=250)

import os

if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
import pygame.event

from utils.csv_handler import csv_handler
from utils.lot_generator import *
import utils.general_utils
from utils.general_utils import action_mapping, dump_arguments, write_stats
from agents.qlearner.qlearn_agent import QLearnerAgent
from utils.feature_extractor import *
from utils.plots.make_plots import plot_all_from_lines
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "RUN_Q_LEARN"


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
         learning_rate=0.01,
         gamma=0.9,
         epsilon=0.3,
         plot_in_training=True,
         plot_interval=100,
         log_rate=10,
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

    agent = QLearnerAgent(sim, time_difference_secs, feature_extractor.input_num,
                          numTraining=n_simulations,
                          epsilon=epsilon,
                          gamma=gamma,
                          alpha=learning_rate,
                          save_folder=save_folder)
    if load_model:
        agent.save_folder = load_folder
        agent.load(load_iter)
        agent.save_folder = save_folder

    result_writer = csv_handler(save_folder, csv_handler.DEFAULT_STATS)
    step = 0
    while True:
        # get old state
        state_old = agent.sim.get_state()

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        state_new, reward, done, results = agent.sim.do_step(*action_mapping[action],
                                                             time_difference_secs)
        step += 1
        if draw_screen and agent.episode % draw_rate == 0:
            pygame.event.pump()
            agent.sim.update_screen()
        iteration_total_reward += reward
        if agent.episode % log_rate == 0 or done:
            write_stats(result_writer, agent.episode,
                        step,
                        reward,
                        iteration_total_reward,
                        results[Results.DISTANCE_TO_TARGET],
                        results[Results.PERCENTAGE_IN_TARGET],
                        results[Results.ANGLE_TO_TARGET],
                        results[Results.SUCCESS],
                        results[Results.COLLISION],
                        done)

        if done:

            # train long memory, plot result
            print(f"Total virtual time: {agent.sim.total_time}")
            print('Simulation', agent.episode, 'Reward', iteration_total_reward, 'Distance',
                  f"{results[Results.DISTANCE_TO_TARGET]:.2f}")
            if agent.episode % log_rate != 0:
                write_stats(result_writer, agent.episode,
                            step,
                            reward,
                            iteration_total_reward,
                            results[Results.DISTANCE_TO_TARGET],
                            results[Results.PERCENTAGE_IN_TARGET],
                            results[Results.ANGLE_TO_TARGET],
                            results[Results.SUCCESS],
                            results[Results.COLLISION],
                            done)

            sim.reset()

            if agent.episode % plot_interval == 0:
                plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

            iteration_total_reward = 0
            if agent.episode >= n_simulations:
                break
            step = 0
            agent.stopEpisode()
            agent.startEpisode()
import os

import torch

if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
import pygame.event

from utils.csv_handler import csv_handler
from utils.lot_generator import *
import utils.general_utils
from utils.general_utils import action_mapping, dump_arguments, write_stats
from agents.genetic.genetic_model import GeneticModel
from utils.feature_extractor import *
from utils.plots.make_plots import plot_all_from_lines
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "RUN_GENETIC"


@dump_arguments(agent_type=AGENT_TYPE)
def main(lot_generator=example_easy,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor9,
         load_model=False,
         load_folder=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=False,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         log_rate=10,
         plot_in_training=True,
         plot_interval=100,
         save_folder=None):
    assert (not load_model) or (load_model and isinstance(load_folder, str))

    i_step = 0
    epochs = 0
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

    model = GeneticModel(feature_extractor.input_num, len(utils.general_utils.action_mapping),
                         save_folder=save_folder)
    if load_model:
        model.change_checkpoint_dir(load_folder)
        model.load_checkpoint()
        model.change_checkpoint_dir(save_folder)

    result_writer = csv_handler(save_folder, csv_handler.DEFAULT_STATS)

    while True:
        i_step += 1
        # get old state
        state_old = sim.get_state()

        # get move
        with torch.no_grad():
            action = torch.argmax(model(torch.FloatTensor([state_old]))).item()

        # perform move and get new state
        state_new, reward, done, results = sim.do_step(*action_mapping[action],
                                                       time_difference_secs)
        if draw_screen and epochs % draw_rate == 0:
            pygame.event.pump()
            sim.update_screen()
        iteration_total_reward += reward

        if i_step % log_rate == 0 or done:
            write_stats(result_writer, epochs,
                        i_step,
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
            print(f"Total virtual time: {sim.total_time}")
            print('Simulation', epochs, 'Reward', iteration_total_reward, 'Distance',
                  f"{results[Results.DISTANCE_TO_TARGET]:.2f}")

            if i_step % log_rate != 0:  # not already printed
                write_stats(result_writer, epochs,
                            i_step,
                            reward,
                            iteration_total_reward,
                            results[Results.DISTANCE_TO_TARGET],
                            results[Results.PERCENTAGE_IN_TARGET],
                            results[Results.ANGLE_TO_TARGET],
                            results[Results.SUCCESS],
                            results[Results.COLLISION],
                            done)
            if (epochs + 1) % plot_interval == 0:
                plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

            sim.reset()
            epochs += 1
            i_step = 0

            iteration_total_reward = 0
            if epochs >= n_simulations:
                break

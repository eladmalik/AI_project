import os

if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
from math import ceil
from tqdm import tqdm
from itertools import count


import torch

from utils.csv_handler import csv_handler
from utils.general_utils import dump_arguments, get_agent_output_folder, action_mapping, write_stats
from utils.lot_generator import *
from utils.plots.make_plots import plot_all_from_lines
from utils.reward_analyzer import *
from agents.dqn2.dqn2_agent import DQNReinforcmentAgent
from agents.dqn2.dqn2_model import DQN_Model
from simulation.simulator import Simulator, DrawingMethod
from utils.feature_extractor import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGENT_TYPE = "RUN_DQN2"


@dump_arguments(agent_type=AGENT_TYPE)
def main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerNew,
         feature_extractor=ExtractorNew,
         load_model=False,
         load_folder=None,
         load_iter=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=False,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         batch_size=256,
         max_memory=200000,
         eps_start=0.05,
         eps_end=0.05,
         eps_decay=500000,
         gamma=0.999,
         log_rate=10,
         plot_in_training=True,
         plot_interval=100,
         target_update=1000,
         save_folder=None):
    iteration_total_reward = 0

    sim = Simulator(lot_generator,
                    reward_analyzer,
                    feature_extractor,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    max_iteration_time_sec=max_iteration_time,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)

    print(f"run device = {device}")

    num_steps = int(ceil(sim.max_simulator_time / time_difference_secs))
    progress_bar = tqdm(range(num_steps))
    progress_bar_iter = progress_bar.__iter__()
    i_episode = 0
    if save_folder is None:
        save_folder = get_agent_output_folder(AGENT_TYPE + '_test_')
    agent = DQNReinforcmentAgent(DQN_Model, feature_extractor,
                                 max_mem=max_memory, batch_size=batch_size, gamma=gamma, eps_start=eps_start,
                                 eps_end=eps_end, eps_decay=eps_decay, target_update=target_update,
                                 save_folder=save_folder, is_eval=False)

    if load_model:
        agent.save_folder = load_folder
        agent.load(load_iter)
        agent.save_folder = save_folder

    result_writer = csv_handler(save_folder, csv_handler.DEFAULT_STATS)

    while True:
        # episodes
        sim.reset()
        i_episode += 1
        state = torch.as_tensor([sim.get_state()], device=device)

        for t in count():
            # Select and perform an action
            action, _ = agent.get_action(state)
            action_item = int(action.item())
            next_state, reward, done, results = sim.do_step(*action_mapping[action_item],
                                                            time_difference_secs)
            next_state = torch.as_tensor([next_state], device=device)

            # Move to the next state
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

            if t % log_rate == 0:
                write_stats(result_writer, i_episode,
                            t,
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
                print('Game', i_episode, 'Reward', iteration_total_reward, 'Distance',
                      f"{results[Results.DISTANCE_TO_TARGET]:.2f}")

                if t % log_rate != 0:  # not already printed
                    write_stats(result_writer, i_episode,
                                t,
                                reward,
                                iteration_total_reward,
                                results[Results.DISTANCE_TO_TARGET],
                                results[Results.PERCENTAGE_IN_TARGET],
                                results[Results.ANGLE_TO_TARGET],
                                results[Results.SUCCESS],
                                results[Results.COLLISION],
                                done)
                if (i_episode + 1) % plot_interval == 0:
                    plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

                progress_bar = tqdm(range(num_steps))
                progress_bar_iter = progress_bar.__iter__()
                iteration_total_reward = 0
                break

        if i_episode > n_simulations:
            print("done")
            break
        sim.reset()


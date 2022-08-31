import os
import sys
import json

import utils.general_utils
from utils.general_utils import dump_arguments
from archived_files.dqn.dqn_model import Agent as DQN_Agent
from agents.ppo.ppo_agent import Agent as PPO_Agent
from utils import calculations
from utils.csv_handler import csv_handler
from utils.enums import StatsType
from utils.lot_generator import *
import utils.feature_extractor
from utils.plots.make_plots import plot_all_from_lines
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod
from agents.ppo_lstm.ppo_lstm_model import PPO_LSTM_Agent
from archived_files.dqn.dqn_train import DQN_Model

from utils.general_utils import action_mapping

FPS = 30
DEBUG = False


@dump_arguments(agent_type="TEST")
def main(
        agent_type: str,
        load_folder: str,
        lot_generator: LotGenerator = generate_lot,
        load_iter: int = None,
        time_difference_secs: float = 0.1,
        max_iteration_time: int = 800,
        draw_screen: bool = True,
        resize_screen: bool = False,
        n_simulations: int = 100,
        plot_in_training: bool = True,
        plot_interval: int = 100,
        save_folder: str = None):
    # initializing the parking lot
    with open(os.path.join(load_folder, utils.general_utils.ARGUMENTS_FILE), "r") as file:
        trained_args = json.load(file)
    feature_extractor_name = trained_args["feature_extractor"]
    feature_extractor_name = feature_extractor_name.split(" ")[1]
    feature_extractor_name = feature_extractor_name.split(".")[-1][:-2]
    feature_extractor = getattr(utils.feature_extractor, feature_extractor_name)
    reward_analyzer = AnalyzerNull

    sim = Simulator(lot_generator, reward_analyzer,
                    feature_extractor,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    max_iteration_time_sec=max_iteration_time,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    total_reward = 0
    successful_parks = 0
    in_parking = 0
    simulations = 1

    agent = None
    if agent_type == "dqn":
        model = DQN_Model(feature_extractor.input_num, len(utils.general_utils.action_mapping), save_folder)
        agent = DQN_Agent(sim, model, save_folder=save_folder)
    elif agent_type == "ppo":
        agent = PPO_Agent(input_dims=feature_extractor.input_num,
                          n_actions=len(utils.general_utils.action_mapping),
                          save_folder=save_folder)
    elif agent_type == "ppo_lstm":
        agent = PPO_LSTM_Agent(feature_extractor.input_num, len(utils.general_utils.action_mapping),
                               save_folder=save_folder)

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

    if agent_type == "ppo_lstm":
        hidden = agent.get_init_hidden()
    while simulations <= n_simulations:
        # The main loop of the simulator. every iteration of this loop equals to one frame in the simulator.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # analyzing the agent's input
        if agent_type == "ppo_lstm":
            action, hidden = agent.get_action(sim.get_state(), hidden)
        else:
            action = agent.get_action(sim.get_state())

        movement, steering = action_mapping[action]

        # performing the input in the simulator
        _, reward, done, results = sim.do_step(movement, steering, time_difference_secs)
        total_reward += reward
        if done:
            result_writer.write_row({
                StatsType.LAST_REWARD: reward,
                StatsType.TOTAL_REWARD: total_reward,
                StatsType.DISTANCE_TO_TARGET: results[Results.DISTANCE_TO_TARGET],
                StatsType.PERCENTAGE_IN_TARGET: results[Results.PERCENTAGE_IN_TARGET],
                StatsType.ANGLE_TO_TARGET: results[Results.ANGLE_TO_TARGET],
                StatsType.SUCCESS: results[Results.SUCCESS],
                StatsType.COLLISION: results[Results.COLLISION]
            })
            if results[Results.SUCCESS]:
                successful_parks += 1
            if results[Results.PERCENTAGE_IN_TARGET] >= 1:
                in_parking += 1

            if simulations % plot_interval == 0:
                plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

            print(f"simulation {simulations}, reward: {total_reward}, success_rate: "
                  f"{(successful_parks / (simulations + 1)):.3f}")

            simulations += 1
            sim.reset()
            total_reward = 0
            if agent_type == "ppo_lstm":
                hidden = agent.get_init_hidden()

        # printing the results:
        if DEBUG:
            print(f"reward: {total_reward:.9f}, done: {done}")
        # updating the screen
        text = {
            "Simulation": f"{simulations + 1}",
            "Velocity": f"{sim.agent.velocity.x:.1f}",
            "Reward": f"{reward:.8f}",
            "Total Reward": f"{total_reward:.8f}",
            "Angle to target": f"{calculations.get_angle_to_target(sim.agent, sim.parking_lot.target_park):.1f}",
            "Success Rate": f"{(successful_parks / (simulations + 1)):.3f}",
            "In Parking Rate": f"{(in_parking / (simulations + 1)):.3f}"
        }
        if draw_screen:
            pygame.event.pump()
            sim.update_screen(text)


if __name__ == '__main__':
    main("ppo_lstm",
         "model/PPO_LSTM_29-08-2022__16-29-40")

import configparser
from copy import copy, deepcopy
import pygame
from tqdm import tqdm

from qlearner.qlearner import QLearnerAgent, ResultsType
from sim.parking_lot import ParkingLot 
from sim.simulator import Simulator
from training_utils.lot_generator import LotGenerator
from training_utils.reward_analyzer import Results
from dqn.trainer import Lot_Generators, Analyzers, Extractors, DrawingMethod, ACTION_TO_MOVEMENT_STEERING


config = configparser.ConfigParser()
config.read("./qlearner/training_settings.ini")
conf_default = config["DEFAULT"]
time_difference = float(conf_default["time_difference"])

if __name__ == '__main__':
    iteration_total_reward = 0
    draw_screen=bool(int(conf_default["draw_screen"]))

    sim = Simulator(Lot_Generators[conf_default["lot_generation"]],
                    Analyzers[conf_default["analyzer"]],
                    Extractors[conf_default["extractor"]],
                    draw_screen=draw_screen,
                    resize_screen=bool(int(conf_default["resize_screen"])),
                    max_iteration_time_sec=int(conf_default["max_iteration_time_sec"]),
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    agent_trainer = QLearnerAgent(sim, time_difference, sim.reward_analyzer, sim.feature_extractor)

    num_steps = int((sim.max_simulator_time // time_difference) + 10)
    progress_bar = tqdm(range(num_steps))
    progress_bar_iter = progress_bar.__iter__()
    iter_num = 0
    game_count = 0
    while True:
        # get old state
        state_old = agent_trainer.sim.get_state()
        lot_old = copy(agent_trainer.sim.parking_lot)

        # get move
        action = agent_trainer.get_action(sim.parking_lot)
       
        # perform move and get new state
        final_movement, final_steering = ACTION_TO_MOVEMENT_STEERING[tuple(action)]
        state_new, reward, done = sim.do_step(final_movement, final_steering,
                                                                time_difference)
        if draw_screen:
            try:
                next(progress_bar_iter)   
            except:
                pass
            if iter_num % 10:
                progress_bar.set_description(f"reward: {reward: .2f}")                                                             
        if draw_screen:
            pygame.event.pump()
            sim.update_screen()
        iteration_total_reward += reward

        results: ResultsType = {Results.COLLISION: sim.is_collision(),
                   Results.PERCENTAGE_IN_TARGET: sim.percentage_in_target_cell(),
                   Results.FRAME: sim.frame,
                   Results.SIMULATION_TIMEOUT: sim.total_time >= sim.max_simulator_time}

        agent_trainer.update(lot_old, results, sim.parking_lot, reward)
        
        if done:
            # train long memory, plot result
            print(f"Total real time: {sim.total_time}")
            final_distance = sim.agent.location.distance_to(sim.parking_lot.target_park.location)

            progress_bar = tqdm(range(num_steps))
            progress_bar_iter = progress_bar.__iter__()
            sim.reset()

          

            print('Game', game_count, 'Reward', iteration_total_reward, 'Distance',
                  f"{final_distance:.2f}")

            iteration_total_reward = 0
            game_count += 1
        
        iter_num += 1
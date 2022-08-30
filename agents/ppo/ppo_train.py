import os
if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
import utils.general_utils
from utils.csv_handler import csv_handler
from utils.enums import DataType
from utils.general_utils import action_mapping, dump_arguments
from utils.lot_generator import *
from utils.reward_analyzer import *
from utils.feature_extractor import *
from utils.plot_maker import plot_all_from_lines
from utils.calculations import get_angle_to_target
from agents.ppo.ppo_agent import Agent
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "PPO"


@dump_arguments(AGENT_TYPE)
def main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor8,
         load_model=False,
         load_folder=None,
         load_iter=None,
         time_difference_secs=0.1,
         max_iteration_time=500,
         draw_screen=True,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         learning_rate=0.0005,
         gamma=0.99,
         lmbda=0.95,
         policy_clip=0.1,
         learn_interval=80,
         batch_size=20,
         n_epochs=4,
         plot_in_training=True,
         plot_interval=100,
         checkpoint_interval=250,
         save_folder=None):
    assert (not load_model) or (load_model and isinstance(load_folder, str))
    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    if save_folder is None:
        save_folder = utils.general_utils.get_agent_output_folder("PPO")
    agent = Agent(n_actions=len(utils.general_utils.action_mapping),
                  batch_size=batch_size,
                  alpha=learning_rate,
                  gamma=gamma,
                  gae_lambda=lmbda,
                  n_epochs=n_epochs,
                  policy_clip=policy_clip,
                  input_dims=tuple([feature_extractor.input_num]),
                  save_folder=save_folder)
    if load_model:
        agent.change_checkpoint_dir(load_folder)
        agent.load_models(load_iter)
        agent.change_checkpoint_dir(save_folder)

    learn_iters = 0
    n_steps = 0
    result_writer = csv_handler(save_folder, [DataType.LAST_REWARD,
                                              DataType.TOTAL_REWARD,
                                              DataType.DISTANCE_TO_TARGET,
                                              DataType.PERCENTAGE_IN_TARGET,
                                              DataType.ANGLE_TO_TARGET,
                                              DataType.SUCCESS,
                                              DataType.COLLISION])

    for i in range(n_simulations):
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        reward = 0
        results = None
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, results = env.do_step(*action_mapping[action], time_difference_secs)
            n_steps += 1
            score += reward
            if draw_screen and i % draw_rate == 0:
                text = {
                    "Run ID": save_folder,
                    "Velocity": f"{env.agent.velocity.x:.1f}",
                    "Reward": f"{reward:.8f}",
                    "Total Reward": f"{score:.8f}",
                    "Angle to target": f"{get_angle_to_target(env.agent, env.parking_lot.target_park):.1f}"
                }
                pygame.event.pump()
                env.update_screen(text)
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % learn_interval == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        result_writer.write_row({
            DataType.LAST_REWARD: reward,
            DataType.TOTAL_REWARD: score,
            DataType.DISTANCE_TO_TARGET: results[Results.DISTANCE_TO_TARGET],
            DataType.PERCENTAGE_IN_TARGET: results[Results.PERCENTAGE_IN_TARGET],
            DataType.ANGLE_TO_TARGET: results[Results.ANGLE_TO_TARGET],
            DataType.SUCCESS: results[Results.SUCCESS],
            DataType.COLLISION: results[Results.COLLISION]
        })
        agent.save_models()
        if (i + 1) % plot_interval == 0:
            plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)
        if (i + 1) % checkpoint_interval == 0:
            agent.save_models(iteration=i)

        print('episode', i, 'score %.9f' % score, 'time_steps', n_steps, 'learning_steps', learn_iters)


if __name__ == '__main__':
    main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor8,
         load_model=False,
         load_folder=None,
         load_iter=None,
         time_difference_secs=0.1,
         max_iteration_time=500,
         draw_screen=True,
         resize_screen=False,
         draw_rate=1,
         n_simulations=100000,
         learning_rate=0.0005,
         gamma=0.99,
         lmbda=0.95,
         policy_clip=0.1,
         learn_interval=80,
         batch_size=20,
         n_epochs=4,
         plot_in_training=True,
         plot_interval=100,
         checkpoint_interval=250)

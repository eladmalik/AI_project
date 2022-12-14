import os

if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
import utils.general_utils
from utils.csv_handler import csv_handler
from utils.general_utils import action_mapping, dump_arguments, write_stats
from utils.lot_generator import *
from utils.reward_analyzer import *
from utils.feature_extractor import *
from utils.plots.make_plots import plot_all_from_lines
from utils.calculations import get_angle_to_target
from agents.ppo.ppo_agent import Agent
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "RUN_PPO"


@dump_arguments(AGENT_TYPE)
def main(lot_generator=example_easy,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor9,
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
         batch_size=20,
         n_epochs=4,
         log_rate=10,
         plot_in_training=True,
         plot_interval=100,
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
                  save_folder=save_folder,
                  is_eval=True)
    if load_model:
        agent.change_checkpoint_dir(load_folder)
        agent.load_models(load_iter)
        agent.change_checkpoint_dir(save_folder)

    learn_iters = 0
    n_steps = 0
    result_writer = csv_handler(save_folder, csv_handler.DEFAULT_STATS)

    for i in range(n_simulations):
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        reward = 0
        results = None
        i_step = 0

        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, results = env.do_step(*action_mapping[action], time_difference_secs)
            n_steps += 1
            i_step += 1
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

            observation = observation_

            if i_step % log_rate == 0 or done:
                write_stats(result_writer, i,
                            i_step,
                            reward,
                            score,
                            results[Results.DISTANCE_TO_TARGET],
                            results[Results.PERCENTAGE_IN_TARGET],
                            results[Results.ANGLE_TO_TARGET],
                            results[Results.SUCCESS],
                            results[Results.COLLISION],
                            done)

        if (i + 1) % plot_interval == 0:
            plot_all_from_lines(result_writer.get_current_data(), save_folder, show=plot_in_training)

        print('episode', i, 'score %.9f' % score, 'time_steps', n_steps, 'learning_steps', learn_iters)

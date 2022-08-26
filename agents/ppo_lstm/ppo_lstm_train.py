# PPO-LSTM
import os
import torch
from torch.distributions import Categorical

from utils.enums import DataType
from agents.ppo_lstm.ppo_lstm_model import PPO_LSTM_Agent
import utils.general_utils
from utils.csv_handler import csv_handler
from utils.general_utils import dump_arguments
from utils.lot_generator import *
from utils.reward_analyzer import *
from utils.feature_extractor import *
from utils.general_utils import action_mapping
from simulation.simulator import Simulator, DrawingMethod

AGENT_TYPE = "PPO_LSTM"


@dump_arguments(agent_type=AGENT_TYPE)
def main(
        lot_generator=generate_lot,
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
        learning_rate=0.0005,
        gamma=0.98,
        lmbda=0.95,
        policy_clip=0.1,
        learn_interval=20,
        n_epochs=2,
        plot_interval=100,
        checkpoint_interval=250,
        save_folder=None):
    assert (not load_model) or (load_model and isinstance(load_folder, str))
    if save_folder is None:
        save_folder = utils.general_utils.get_agent_output_folder(AGENT_TYPE)
    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)

    observation_space_size = env.feature_extractor.input_num
    action_space_size = len(utils.general_utils.action_mapping)

    agent = PPO_LSTM_Agent(observation_space_size, action_space_size, save_folder=save_folder,
                           lr=learning_rate,
                           gamma=gamma,
                           lmbda=lmbda,
                           eps_clip=policy_clip,
                           n_epochs=n_epochs,
                           learn_interval=learn_interval)
    if load_model:
        agent.save_folder = load_folder
        agent.load()
        agent.save_folder = save_folder
    score = 0.0
    result_writer = csv_handler(save_folder, [DataType.LAST_REWARD,
                                              DataType.TOTAL_REWARD,
                                              DataType.DISTANCE_TO_TARGET,
                                              DataType.PERCENTAGE_IN_TARGET,
                                              DataType.ANGLE_TO_TARGET,
                                              DataType.SUCCESS,
                                              DataType.COLLISION])

    for n_epi in range(n_simulations):
        h_out = agent.get_init_hidden()
        state = env.reset()
        done = False
        reward = 0
        results = None

        while not done:
            for t in range(learn_interval):
                h_in = h_out
                prob, h_out = agent.pi(torch.Tensor(state).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                action = m.sample().item()
                s_prime, reward, done, results = env.do_step(*action_mapping[action], time_difference_secs)
                score += reward
                if draw_screen and n_epi % draw_rate == 0:
                    text = {
                        "Run folder": save_folder,
                        "Velocity": f"{env.agent.velocity.x:.1f}",
                        "Reward": f"{reward:.8f}",
                        "Total Reward": f"{score:.8f}",
                        "Angle to target": f""
                                           f"{get_angle_to_target(env.agent, env.parking_lot.target_park):.1f}",
                        "Percentage in target": f"{results[Results.PERCENTAGE_IN_TARGET]:.4f}"
                    }
                    pygame.event.pump()
                    env.update_screen(text)

                agent.put_data(
                    (state, action, reward / 100.0, s_prime, prob[action].item(), h_in, h_out, done))
                state = s_prime
                if done:
                    break

            agent.train_net()

        result_writer.write_row({
            DataType.LAST_REWARD: reward,
            DataType.TOTAL_REWARD: score,
            DataType.DISTANCE_TO_TARGET: results[Results.DISTANCE_TO_TARGET],
            DataType.PERCENTAGE_IN_TARGET: results[Results.PERCENTAGE_IN_TARGET],
            DataType.ANGLE_TO_TARGET: results[Results.ANGLE_TO_TARGET],
            DataType.SUCCESS: results[Results.SUCCESS],
            DataType.COLLISION: results[Results.COLLISION]
        })
        if (n_epi + 1) % checkpoint_interval == 0:
            agent.save(custom_name=f"iter_{n_epi}.pth")
        agent.save()
        print("# of episode :{}, score : {:.1f}".format(n_epi, score))
        score = 0.0


if __name__ == '__main__':
    main()

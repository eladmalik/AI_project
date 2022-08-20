import os.path

import utils
from training_utils.lot_generator import *
from training_utils.reward_analyzer import *
from training_utils.feature_extractor import *
import numpy as np

from sim.car import Movement, Steering
from ppo_agent import Agent, ACTOR_PTH_NAME, CRITIC_PTH_NAME
from sim.simulator import Simulator, DrawingMethod

action_mapping = {
    0: (Movement.NEUTRAL, Steering.NEUTRAL),
    1: (Movement.NEUTRAL, Steering.LEFT),
    2: (Movement.NEUTRAL, Steering.RIGHT),
    3: (Movement.FORWARD, Steering.NEUTRAL),
    4: (Movement.FORWARD, Steering.LEFT),
    5: (Movement.FORWARD, Steering.RIGHT),
    6: (Movement.BACKWARD, Steering.NEUTRAL),
    7: (Movement.BACKWARD, Steering.LEFT),
    8: (Movement.BACKWARD, Steering.RIGHT),
    9: (Movement.BRAKE, Steering.NEUTRAL),
    10: (Movement.BRAKE, Steering.LEFT),
    11: (Movement.BRAKE, Steering.RIGHT)
}


def get_agent_output_folder():
    folder = os.path.join("model", f'PPO_{utils.get_time()}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD MODEL HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    load_model = False
    model_folder = os.path.join("model", "PPO_18-08-2022__19-27-30")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHANGE HYPER-PARAMETERS HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    lot_generator = example0
    reward_analyzer = AnalyzerNew
    feature_extractor = ExtractorNew
    time_difference_secs = 0.1
    max_iteration_time = 20
    draw_screen = True
    draw_rate = 1

    N = 20
    batch_size = 5
    n_epochs = 4
    policy_clip = 0.1
    alpha = 0.001
    n_games = 10000

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=False,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    save_folder = get_agent_output_folder()
    agent = Agent(n_actions=12, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, policy_clip=policy_clip,
                  input_dims=tuple([feature_extractor.input_num]),
                  save_folder=save_folder)
    if load_model:
        agent.change_checkpoint_dir(model_folder)
        agent.load_models()
        agent.change_checkpoint_dir(save_folder)

    best_score = -float("inf")
    score_history = []
    mean_score_history = []
    distance_history = []
    mean_distance_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.do_step(action_mapping[action][0], action_mapping[action][1],
                                                     time_difference_secs)
            n_steps += 1
            score += reward
            if draw_screen and i % draw_rate == 0:
                pygame.event.pump()
                env.update_screen()
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        
        agent.learn()

        score_history.append(score)
        mean_score_history.append(sum(score_history) / len(score_history))
        avg_score = np.mean(score_history[-100:])
        distance_history.append(env.agent.location.distance_to(env.parking_lot.target_park.location))
        mean_distance_history.append(sum(distance_history) / len(distance_history))
        if (i + 1) % 200 == 0:
            utils.plot_rewards(score_history, mean_score_history, save_folder)
            utils.plot_distances(distance_history, mean_distance_history, save_folder)
            agent.save_models()

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]

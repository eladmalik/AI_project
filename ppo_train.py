import os.path
import lot_generator
from reward_analyzer import *
from feature_extractor import *
import numpy as np

from car import Movement, Steering
from ppo_agent import Agent
from simulator import Simulator, DrawingMethod

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

if __name__ == '__main__':
    lot_generator = lot_generator.example0
    reward_analyzer = AnalyzerAccumulating4
    feature_extractor = Extractor4
    draw_screen = True
    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=500,
                    draw_screen=draw_screen,
                    resize_screen=False,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    time_difference_secs = 0.1
    N = 5
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=12, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=tuple([feature_extractor.input_num]))
    n_games = 10000

    figure_file = os.path.join('plots', "cartpole.png")

    best_score = -float("inf")
    score_history = []

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
            if draw_screen:
                # print(f"action: {action}")
                pygame.event.pump()
                env.update_screen()
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]

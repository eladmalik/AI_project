# PPO-LSTM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import calculations
import utils
from car import Movement, Steering
from lot_generator import *
from reward_analyzer import *
from feature_extractor import *
import time
import numpy as np

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
    folder = os.path.join("model", f'PPO_LSTM2_{utils.get_time()}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD MODEL HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


load_model = True
model_folder = os.path.join("model", "PPO_LSTM2_21-08-2022__18-11-38")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHANGE HYPER-PARAMETERS HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from simulator import Simulator, DrawingMethod

lot_generator = example2
reward_analyzer = AnalyzerAccumulating4FrontBack
feature_extractor = Extractor7
time_difference_secs = 0.1
max_iteration_time = 800
draw_screen = True
draw_rate = 1

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 2
T_horizon = 20


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class PPO(nn.Module):
    def __init__(self, input_num, action_num, save_folder="tmp/ppo_lstm2"):
        super(PPO, self).__init__()
        self.data = []
        self.lstm_input = 128

        self.fc1 = nn.Linear(input_num, 128)
        self.fc2 = nn.Linear(128, self.lstm_input)
        self.lstm = nn.LSTM(self.lstm_input, 32)
        self.fc_pi = nn.Linear(32, action_num)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.save_folder = save_folder

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 1, self.lstm_input)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 1, self.lstm_input)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst,
                                                                                dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(
            prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(), os.path.join(self.save_folder, "agent.pth"))

    def load(self):
        self.load_state_dict(torch.load(os.path.join(self.save_folder, "agent.pth")))


def main():
    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=False,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)

    observation_space_size = env.feature_extractor.input_num
    action_space_size = 12
    save_folder = get_agent_output_folder()
    model = PPO(observation_space_size, action_space_size, save_folder=save_folder)
    if load_model:
        model.save_folder = model_folder
        model.load()
        model.save_folder = save_folder
    score = 0.0
    plot_interval = 10
    distance_history = []
    mean_distance_history = []

    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.Tensor(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done = env.do_step(*action_mapping[a], time_difference_secs)
                score += r
                if draw_screen:
                    text = {
                        "Velocity": f"{env.agent.velocity.x:.1f}",
                        "Reward": f"{r:.8f}",
                        "Total Reward": f"{score:.8f}",
                        "Angle to target": f"{calculations.get_angle_to_target(env.agent, env.parking_lot.target_park):.1f}"
                    }
                    pygame.event.pump()
                    env.update_screen(text)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime
                if done:
                    break

            model.train_net()

        distance_history.append(env.agent.location.distance_to(env.parking_lot.target_park.location))
        mean_distance_history.append(sum(distance_history)/len(distance_history))
        if n_epi % plot_interval == 0 and n_epi != 0:
            utils.plot_distances(distance_history, mean_distance_history, save_folder)
        model.save()
        print("# of episode :{}, score : {:.1f}".format(n_epi, score))
        score = 0.0


if __name__ == '__main__':
    main()

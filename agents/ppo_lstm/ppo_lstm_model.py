import os

import torch
from torch import nn as nn, optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F
import pickle

MODEL_STRUCT = "structure.pickle"


class PPO_Agent(nn.Module):
    def __init__(self, input_num, action_num, save_folder="tmp/ppo_lstm",
                 lr=0.0005,
                 gamma=0.98,
                 lmbda=0.95,
                 eps_clip=0.1,
                 n_epochs=2,
                 learn_interval=20):
        super(PPO_Agent, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.learn_interval = learn_interval

        self.data = []
        self.lstm_input = 128
        self.lstm_output = 32

        self.pre_lstm = nn.Sequential(nn.Linear(input_num, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, self.lstm_input),
                                      nn.ReLU())
        self.lstm = nn.LSTM(self.lstm_input, self.lstm_output)
        self.fc_pi = nn.Linear(self.lstm_output, action_num)
        self.fc_v = nn.Linear(self.lstm_output, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.save_folder = save_folder
        with open(os.path.join(self.save_folder, MODEL_STRUCT), "wb") as file:
            pickle.dump((self.lstm_input, self.lstm_output, self.pre_lstm, self.lstm, self.fc_pi, self.fc_v,
                         self.optimizer), file)

    def pi(self, x, hidden):
        x = self.pre_lstm(x)
        x = x.view(-1, 1, self.lstm_input)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = self.pre_lstm(x)
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

        for i in range(self.n_epochs):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

    def get_init_hidden(self):
        return torch.zeros([1, 1, self.lstm_output], dtype=torch.float), \
               torch.zeros([1, 1, self.lstm_output], dtype=torch.float)

    def get_action(self, state, hidden):
        prob, h_out = self.pi(torch.Tensor(state).float(), hidden)
        prob = prob.view(-1)
        m = Categorical(prob)
        action = m.sample().item()
        return action, h_out

    def save(self):
        torch.save(self.state_dict(), os.path.join(self.save_folder, "agent.pth"))

    def load(self):
        if os.path.exists(os.path.join(self.save_folder, MODEL_STRUCT)):
            with open(os.path.join(self.save_folder, MODEL_STRUCT), "rb") as file:
                self.lstm_input, self.lstm_output, self.pre_lstm, self.lstm, self.fc_pi, self.fc_v, \
                self.optimizer = pickle.load(file)
        self.load_state_dict(torch.load(os.path.join(self.save_folder, "agent.pth")))

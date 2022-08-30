# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import os
import pickle
from typing import Any, Dict, List
from utils.general_utils import action_mapping
from simulation.simulator import Simulator
from utils.reward_analyzer import Results

import numpy as np
import random

ResultsType = Dict[Results, Any]

WEIGHTS_FILE_NAME = "weights.pickle"


class QLearnerAgent:
    """
     ApproximateQLearningAgent
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
    """

    def __init__(self, simulator: Simulator, time: float, input_num: int, numTraining=100, epsilon=0.5,
                 alpha=0.5, gamma=0.9,
                 save_folder="tmp"):

        # You might want to initialize weights here.
        self.__w = np.random.rand(input_num)
        self.discount = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numTraining = numTraining
        self.episodeRewards = 0
        self.save_folder = save_folder
        self.episode = 1

        self.sim = simulator
        self.time = time

    def getQValue(self, state: List[float], action: int) -> float:
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        new_state = self.sim.peek_step(*action_mapping[action], self.time)[0]
        return np.dot(new_state, self.__w).item()

    def getLegalActions(self, state: List[float]) -> List[int]:
        return list(action_mapping.keys())

    def getPolicy(self, state: List[float]):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        actions = self.getLegalActions(state)
        random.shuffle(actions)
        values = np.array(tuple(self.getQValue(state, a) for a in actions))
        return actions[np.argmax(values)]

    def get_action(self, state: List[float]):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)

        if np.random.random() < self.epsilon:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def getValue(self, state: List[float]):
        actions = self.getLegalActions(state)
        return max(self.getQValue(state, a) for a in actions) if len(actions) else 0.0

    def update(self, state: List[float], action: int, nextState: List[float], reward: float):
        """
        Should update your weights based on transition
        """
        corrections = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for i in range(len(state)):
            self.__w[i] += self.alpha * corrections * state[i]

    def observeTransition(self, state: List[float], action: int, nextState: List[float],
                          deltaReward: float):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        self.episodeRewards = 0

    def stopEpisode(self):
        self.episode += 1

    def save(self, iteration=None):
        name = WEIGHTS_FILE_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        with open(os.path.join(self.save_folder, WEIGHTS_FILE_NAME), "wb") as file:
            pickle.dump(self.__w, file)

    def load(self, iteration=None):
        name = WEIGHTS_FILE_NAME
        if iteration is not None:
            name += f"_iter_{iteration}.pth"
        with open(os.path.join(self.save_folder, WEIGHTS_FILE_NAME), "rb") as file:
            self.__w = pickle.load(file)

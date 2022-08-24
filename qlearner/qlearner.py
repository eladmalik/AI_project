# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from typing import Any, Dict, List
from sim.car import Movement, Steering
from sim.parking_lot import ParkingLot
from sim.simulator import Simulator
from training_utils.feature_extractor import FeatureExtractor
from training_utils.reward_analyzer import Results, RewardAnalyzer
from dqn.trainer import MOVEMENT_STEERING_TO_ACTION, ACTION_TO_MOVEMENT_STEERING, ActionType

import numpy as np
import random

ResultsType = Dict[Results, Any]


class QLearnerAgent:
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
    """

    def __init__(self, simulator: Simulator, time: float, reward_analyzer: RewardAnalyzer, extractor: FeatureExtractor, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        self.featExtractor = extractor
        
        # You might want to initialize weights here.
        self.reward_analyzer = reward_analyzer
        self.extractor = extractor
        self.__w = np.random.rand(extractor.input_num)
        self.discount = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numTraining = numTraining
        self.episodeRewards = 0

        self.sim = simulator
        self.time = time


    def getQValue(self, state: ParkingLot, results: Dict[Results, Any]) -> float:
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        return np.dot(self.extractor.get_state(state), self.__w)

    def getLegalActions(self, state: ParkingLot) -> List[ActionType]:
        return list(MOVEMENT_STEERING_TO_ACTION.values())

    def getPolicy(self, state: ParkingLot):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        actions = self.getLegalActions(state)
        random.shuffle(actions)
        values = np.array(tuple(self.getQValue(state, self.sim.simulate_step(*ACTION_TO_MOVEMENT_STEERING[tuple(a)], self.time)[3]) for a in actions))
        return actions[np.argmax(values)]

    def get_action(self, state: ParkingLot):
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

    def getValue(self, state: ParkingLot):
        actions = self.getLegalActions(state)
        simulations = ((*self.sim.simulate_step(*ACTION_TO_MOVEMENT_STEERING[tuple(a)], self.time), a) for a in actions)
        return max(self.getQValue(self.sim.parking_lot, simulation[3]) for simulation in simulations) if len(actions) else 0.0

    def update(self, state: ParkingLot, results: ResultsType, nextState: ParkingLot, reward: float):
        """
        Should update your weights based on transition
        """
        features = self.extractor.get_state(state)
        corrections = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, results)
        for i in range(len(features)):
            self.__w[i] += self.alpha * corrections * features[i]

    def observeTransition(self, state: ParkingLot,action: ResultsType, nextState: ParkingLot, deltaReward: float):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same argumentss
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        self.episodeRewards = 0

    def stopEpisode(self):
        pass

    def train(self, state: ParkingLot, action: ResultsType, nextState: ParkingLot, deltaReward: float):
        """
        Called by Pacman game at the terminal state
        """
        deltaReward = self.getQValue(nextState, action) - self.getQValue(state, action)
        self.observeTransition(state, action, nextState, deltaReward)
        self.stopEpisode()

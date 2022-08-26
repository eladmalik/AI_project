import argparse
import os
if __name__ == '__main__':
    os.chdir("..")
from utils.lot_generator import *
from utils.reward_analyzer import *
from utils.feature_extractor import *


def parse_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="The type of agent to train", choices=["dqn", "ppo", "ppo_lstm",
                                                                             "genetic"])
    parser.add_argument("--lot", help="lot generator function", default="generate_lot", type=str)
    parser.add_argument("--reward", help="reward analyzer class",
                        default="AnalyzerAccumulating4FrontBack", type=str)
    parser.add_argument("--feature", help="feature extractor class", default="Extractor8", type=str)
    parser.add_argument("--load", help="the folder to load the model from", default=None, type=str)
    parser.add_argument("--time", help="time interval between actions", default=0.1, type=float)
    parser.add_argument("--max_time", help="maximum virtual time for a single simulation", default=800,
                        type=int)
    parser.add_argument("-d", help="draw the simulation", action='store_true')
    parser.add_argument("-r", help="resize the simulation screen to the lot size", action='store_true')
    parser.add_argument("--draw_rate", help="how many iterations a simulation will be drawn", default=1,
                        type=int)
    parser.add_argument("--n", help="how many simulations to run", default=1000,type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(parse_train_arguments())

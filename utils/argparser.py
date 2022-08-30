import argparse
import json
import os
import inspect
from warnings import warn

if __name__ == '__main__':
    os.chdir("..")
import utils.lot_generator
import utils.reward_analyzer
import utils.feature_extractor
import utils.general_utils
import agents.dqn.dqn_train
import agents.dqn.dqn_run
import agents.ppo.ppo_train
import agents.ppo.ppo_run
import agents.dqn2.dqn2_train
import agents.dqn2.dqn2_run
import agents.ppo_lstm.ppo_lstm_train
import agents.ppo_lstm.ppo_lstm_run
import agents.genetic.genetic_train
import agents.qlearner.qlearn_train
import agents.qlearner.qlearn_run
from utils.reward_analyzer import AnalyzerNull
from utils.general_utils import isfloat, isint

train_functions = {
    "qlearn": agents.qlearner.qlearn_train.main,
    "dqn": agents.dqn.dqn_train.main,
    "dqn2": agents.dqn2.dqn2_train.main,
    "ppo": agents.ppo.ppo_train.main,
    "ppo_lstm": agents.ppo_lstm.ppo_lstm_train.main,
    "genetic": agents.genetic.genetic_train.main
}

run_functions = {
    "qlearn": agents.qlearner.qlearn_run.main,
    "dqn": agents.dqn.dqn_run.main,
    "dqn2": agents.dqn2.dqn2_run.main,
    "ppo": agents.ppo.ppo_run.main,
    "ppo_lstm": agents.ppo_lstm.ppo_lstm_run.main
}


def _get_default_parameters(run_function):
    relevant_args = set(inspect.signature(run_function).parameters.keys())
    return {arg: inspect.signature(run_function).parameters[arg].default for arg in relevant_args}


def parse_train_arguments():
    """
    Parses the arguments needed to train the agents.
    returns the function to run and its argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="The type of agent to train", choices=["dqn", "dqn2", "ppo", "ppo_lstm",
                                                                             "genetic", "qlearn"])
    parser.add_argument("--lot", help="lot generator function", type=str,
                        metavar="{parking lot generator}")
    parser.add_argument("--reward", help="{reward analyzer class}", type=str, metavar="{reward analyzer "
                                                                                      "class}")
    parser.add_argument("--feature", help="feature extractor class", type=str,
                        metavar="{feature extractor class}")
    parser.add_argument("--load", help="the folder to load the model from", type=str,
                        metavar="{model folder}")
    parser.add_argument("--load_iter", help="the iteration number to load. do not specify to load the last "
                                            "checkpoint", type=int, metavar="{iteration number}")
    parser.add_argument("--time", help="time interval between actions", type=float,
                        metavar="{time interval}")
    parser.add_argument("--max_time", help="maximum virtual time for a single simulation",
                        type=int, metavar="{max virtual simulation time}")
    parser.add_argument("-d", help="draw the simulation. Setting this to true will reduce performance",
                        action='store_true')
    parser.add_argument("-r",
                        help="resize the simulation screen to the lot size. Setting this to true"
                             " will reduce performance", action='store_true')
    parser.add_argument("--draw_rate", help="how many iterations a simulation will be drawn",
                        type=int, metavar="{simulation draw rate}")
    parser.add_argument("--n", help="how many simulations to run", type=int,
                        metavar="{number of simulations}")
    parser.add_argument("--lr", help="learning rate", type=float, metavar="{learning rate}")
    parser.add_argument("--g", help="discount rate", type=float, metavar="{discount rate}")
    parser.add_argument("--eps", help="epsilon for random action. used in the Q-Learner. 0 means no "
                                      "randomness, 1 means always random", type=float,
                        metavar="{epsilon}")
    parser.add_argument("--lm", help="smoothing parameter for PPO", type=float,
                        metavar="{smoothing parameter}")
    parser.add_argument("--clip", help="epsilon for PPO policy clip", type=float,
                        metavar="{policy clip epsilon}")
    parser.add_argument("--n_learn", help="learn interval for PPO - how many steps should pass before "
                                          "learning", type=int, metavar="{learn interval}")
    parser.add_argument("--n_epoch", help="for PPO. how many times the current data should be learned",
                        type=int, metavar="{epochs num}")
    parser.add_argument("--batch", help="batch size", type=int, metavar="{batch size}")
    parser.add_argument("-p", help="use if showing the plots is wanted", action='store_true')
    parser.add_argument("--p_rate", help="how many simulations should pass until new plots are drawn",
                        type=int, metavar="{plot rate}")
    parser.add_argument("--c_rate",
                        help="how many simulations should pass until a new separate model state is saved",
                        type=int, metavar="{checkpoint rate}")

    parser.add_argument("--n_gen", help="Number of generations in genetic training", type=int,
                        metavar="{generations num}")
    parser.add_argument("--n_mate", help="Number of parents mating in each generation for genetic training",
                        type=int, metavar="{parents mating num}")
    parser.add_argument("--mutate_percent",
                        help="Percent of genes mutating in each new solution in genetic training",
                        type=int, metavar="{mutation percent}")
    parser.add_argument("--n_parents_keep",
                        help="Number of parents to keep in the next population in genetic training. -1 "
                             "means keep all parents and 0 means keep nothing.", type=int,
                        metavar="{num of parents to keep}")
    args = parser.parse_args()

    kwargs = {
        "lot_generator": getattr(utils.lot_generator, args.lot) if args.lot is not None else None,
        "reward_analyzer": getattr(utils.reward_analyzer, args.reward) if args.reward is not None else None,
        "feature_extractor": getattr(utils.feature_extractor, args.feature) if args.feature is not None else
        None,
        "time_difference_secs": args.time,
        "max_iteration_time": args.max_time,
        "load_model": args.load is not None,
        "load_folder": args.load,
        "load_iter": args.load_iter,
        "draw_screen": args.d,
        "resize_screen": args.r,
        "draw_rate": args.draw_rate,
        "n_simulations": args.n,
        "learning_rate": args.lr,
        "gamma": args.g,
        "epsilon": args.eps,
        "lmbda": args.lm,
        "policy_clip": args.clip,
        "learn_interval": args.n_learn,
        "n_epochs": args.n_epoch,
        "batch_size": args.batch,
        "plot_interval": args.p_rate,
        "plot_in_training": args.p,
        "checkpoint_interval": args.c_rate,
        "num_generations": args.n_gen,
        "num_parents_mating": args.n_mate,
        "genes_mutation_percent": args.mutate_percent,
        "num_parents_to_keep": args.n_parents_keep
    }

    train_function = train_functions[args.agent]
    relevant_args = set(inspect.signature(train_function).parameters.keys())
    for arg in kwargs:
        if kwargs[arg] is not None and arg not in relevant_args:
            warn(
                f"parameter {str(arg)} is specified, but {str(args.agent)} doesn't use this argument",
                UserWarning)

    default_params = _get_default_parameters(train_function)
    for arg in kwargs:
        if kwargs[arg] is None and arg in default_params:
            kwargs[arg] = default_params[arg]

    kwargs = {key: kwargs[key] for key in relevant_args if key in kwargs.keys()}
    return train_function, kwargs


def parse_run_arguments():
    """
    Parses the arguments needed to train the agents.
    returns the function to run and its argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="The type of agent to train", choices=["dqn", "dqn2", "ppo",
                                                                             "ppo_lstm", "qlearn"])
    parser.add_argument("load", help="the folder to load the model from", type=str,
                        metavar="{model folder}")
    parser.add_argument("--lot", help="lot generator function", type=str,
                        metavar="{parking lot generator}")
    # feature extractor is loaded from the loaded model
    parser.add_argument("--load_iter", help="the iteration number to load. do not specify to load the last "
                                            "checkpoint", type=int, metavar="{iteration number}")
    parser.add_argument("--time", help="time interval between actions", type=float,
                        metavar="{time interval}")
    parser.add_argument("--max_time", help="maximum virtual time for a single simulation",
                        type=int, metavar="{max virtual simulation time}")
    parser.add_argument("-d", help="draw the simulation. Setting this to true will reduce performance",
                        action='store_true')
    parser.add_argument("-r",
                        help="resize the simulation screen to the lot size. Setting this to true"
                             " will reduce performance", action='store_true')
    parser.add_argument("--draw_rate", help="how many iterations a simulation will be drawn",
                        type=int, metavar="{simulation draw rate}")
    parser.add_argument("--n", help="how many simulations to run", type=int,
                        metavar="{number of simulations}")
    parser.add_argument("-p", help="use if showing the plots is wanted", action='store_true')
    parser.add_argument("--p_rate", help="how many simulations should pass until new plots are drawn",
                        type=int, metavar="{plot rate}")
    args = parser.parse_args()

    kwargs = {
        "lot_generator": getattr(utils.lot_generator, args.lot) if args.lot is not None else None,
        "time_difference_secs": args.time,
        "max_iteration_time": args.max_time,
        "load_model": True,
        "load_folder": args.load,
        "load_iter": args.load_iter,
        "draw_screen": args.d,
        "resize_screen": args.r,
        "draw_rate": args.draw_rate,
        "n_simulations": args.n,
        "plot_interval": args.p_rate,
        "plot_in_training": args.p,
    }
    with open(os.path.join(args.load, utils.general_utils.ARGUMENTS_FILE), "r") as file:
        trained_args = json.load(file)
    feature_extractor_name = trained_args["feature_extractor"]
    feature_extractor_name = feature_extractor_name.split(" ")[1]
    feature_extractor_name = feature_extractor_name.split(".")[-1][:-2]
    feature_extractor = getattr(utils.feature_extractor, feature_extractor_name)
    reward_analyzer = AnalyzerNull
    kwargs["feature_extractor"] = feature_extractor
    kwargs["reward_analyzer"] = reward_analyzer
    for org_arg in trained_args:
        if org_arg not in kwargs and org_arg != "load_iter":
            if isint(trained_args[org_arg]):
                kwargs[org_arg] = int(trained_args[org_arg])
            elif isfloat(trained_args[org_arg]):
                kwargs[org_arg] = float(trained_args[org_arg])
            else:
                kwargs[org_arg] = trained_args[org_arg]

    run_function = run_functions[args.agent]
    relevant_args = set(inspect.signature(run_function).parameters.keys())
    # for arg in kwargs:
    #     if kwargs[arg] is not None and arg not in relevant_args:
    #         warn(
    #             f"parameter {str(arg)} is specified, but {str(args.agent)} doesn't use this argument",
    #             UserWarning)

    default_params = _get_default_parameters(run_function)
    for arg in kwargs:
        if kwargs[arg] is None and arg in default_params:
            kwargs[arg] = default_params[arg]

    kwargs = {key: kwargs[key] for key in relevant_args if key in kwargs.keys()}
    return run_function, kwargs


if __name__ == '__main__':
    parse_run_arguments()
    a = 1

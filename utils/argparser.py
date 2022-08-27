import argparse
import os
import inspect

if __name__ == '__main__':
    os.chdir("..")
import utils.lot_generator
import utils.reward_analyzer
import utils.feature_extractor
import agents.dqn.dqn_train
import agents.ppo.ppo_train
import agents.ppo_lstm.ppo_lstm_train
import agents.genetic.genetic_train

run_functions = {
    "dqn": agents.dqn.dqn_train.main,
    "ppo": agents.ppo.ppo_train.main,
    "ppo_lstm": agents.ppo_lstm.ppo_lstm_train.main,
    "genetic": agents.genetic.genetic_train.main
}


def parse_train_arguments():
    """
    Parses the arguments needed to train the agents.
    returns the function to run and its argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="The type of agent to train", choices=["dqn", "ppo", "ppo_lstm",
                                                                             "genetic"])
    parser.add_argument("--lot", help="lot generator function", default="generate_lot", type=str,
                        metavar="{parking lot generator}")
    parser.add_argument("--reward", help="{reward analyzer class}",
                        default="AnalyzerAccumulating4FrontBack", type=str, metavar="{reward analyzer class}")
    parser.add_argument("--feature", help="feature extractor class", default="Extractor8", type=str,
                        metavar="{feature extractor class}")
    parser.add_argument("--load", help="the folder to load the model from", default=None, type=str,
                        metavar="{model folder}")
    parser.add_argument("--load_iter", help="the iteration number to load. do not specify to load the last "
                                            "checkpoint", type=int, metavar="{iteration number}")
    parser.add_argument("--time", help="time interval between actions", default=0.1, type=float,
                        metavar="{time interval}")
    parser.add_argument("--max_time", help="maximum virtual time for a single simulation", default=800,
                        type=int, metavar="{max virtual simulation time}")
    parser.add_argument("-d", help="draw the simulation. Setting this to true will reduce performance",
                        action='store_true')
    parser.add_argument("-r",
                        help="resize the simulation screen to the lot size. Setting this to true"
                             " will reduce performance", action='store_true')
    parser.add_argument("--draw_rate", help="how many iterations a simulation will be drawn", default=1,
                        type=int, metavar="{simulation draw rate}")
    parser.add_argument("--n", help="how many simulations to run", default=1000, type=int,
                        metavar="{number of simulations}")
    parser.add_argument("--lr", help="learning rate", default=-1, type=float, metavar="{learning rate}")
    parser.add_argument("--g", help="discount rate", default=0.99, type=float, metavar="{discount rate}")
    parser.add_argument("--lm", help="smoothing parameter for PPO", default=0.95, type=float,
                        metavar="{smoothing parameter}")
    parser.add_argument("--clip", help="epsilon for PPO policy clip", default=0.1, type=float,
                        metavar="{policy clip epsilon}")
    parser.add_argument("--n_learn", help="learn interval for PPO - how many steps should pass before "
                                          "learning", default=-1, type=int, metavar="{learn interval}")
    parser.add_argument("--n_epoch", help="for PPO. how many times the current data should be learned",
                        default=2, type=int, metavar="{epochs num}")
    parser.add_argument("--batch", help="batch size", default=-1, type=int, metavar="{batch size}")
    parser.add_argument("-p", help="use if showing the plots is wanted", action='store_true')
    parser.add_argument("--p_rate", help="how many simulations should pass until new plots are drawn",
                        default=100, type=int, metavar="{plot rate}")
    parser.add_argument("--c_rate",
                        help="how many simulations should pass until a new separate model state is saved",
                        default=250, type=int, metavar="{checkpoint rate}")

    parser.add_argument("--n_gen", help="Number of generations in genetic training", default=100, type=int,
                        metavar="{generations num}")
    parser.add_argument("--n_mate", help="Number of parents mating in each generation for genetic training",
                        default=5, type=int, metavar="{parents mating num}")
    parser.add_argument("--mutate_percent",
                        help="Percent of genes mutating in each new solution in genetic training",
                        default=10, type=int, metavar="{mutation percent}")
    parser.add_argument("--n_parents_keep",
                        help="Number of parents to keep in the next population in genetic training. -1 "
                             "means keep all parents and 0 means keep nothing.", default=-1, type=int,
                        metavar="{num of parents to keep}")
    args = parser.parse_args()

    kwargs = {
        "lot_generator": getattr(utils.lot_generator, args.lot),
        "reward_analyzer": getattr(utils.reward_analyzer, args.reward),
        "feature_extractor": getattr(utils.feature_extractor, args.feature),
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

    run_function = run_functions[args.agent]
    if args.agent == "dqn":
        if args.lr == -1:
            kwargs["learning_rate"] = 0.001
        if args.batch == -1:
            kwargs["batch_size"] = 1000
    elif args.agent == "ppo":
        if args.lr == -1:
            kwargs["learning_rate"] = 0.0005
        if args.batch == -1:
            kwargs["batch_size"] = 20
        if args.n_learn == -1:
            kwargs["learn_interval"] = 80
    else:
        if args.lr == -1:
            kwargs["learning_rate"] = 0.0005
        if args.batch == -1:
            kwargs["batch_size"] = 20
        if args.n_learn == -1:
            kwargs["learn_interval"] = 20

    relevant_args = set(inspect.signature(run_function).parameters.keys())
    kwargs = {key: kwargs[key] for key in relevant_args if key in kwargs.keys()}
    return run_function, kwargs

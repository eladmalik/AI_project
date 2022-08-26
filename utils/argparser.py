import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="The type of agent to train", choices=["dqn", "ppo", "ppo_lstm",
                                                                             "genetic"])
    parser.add_argument("--")

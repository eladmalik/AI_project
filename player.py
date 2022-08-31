from utils.argparser import parse_player_arguments
from agents.manual.manual_run import main

if __name__ == '__main__':
    kwargs = parse_player_arguments()
    main(**kwargs)

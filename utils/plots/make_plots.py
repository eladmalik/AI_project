import os
from typing import Dict, List, Any

from utils.enums import StatsType
from utils.plots.plot_types.genetic_plots import *
from utils.plots.plot_types.reinforcement_plots import *
from utils.plots.plot_types.compare_plots import *
from utils.plots.plot_utils import arrange_data, get_only_dones
from utils.csv_handler import CSV_FILE, csv_handler


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REINFORCEMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_all(data_dict: Dict[StatsType, List[Any]], save_folder: str, last_epochs: int = 100, show=False,
             start=0):
    plot_distance_data(data_dict, save_folder, last_epochs, show, start)
    plot_percentage_in_target_data(data_dict, save_folder, last_epochs, show, start)
    plot_success_collision_rate(data_dict, save_folder, last_epochs, show, start)


def plot_all_from_lines(lines: List[List[Any]], save_folder: str, last_epochs: int = 100, show=False,
                        start=0, end=-1):
    data_dict = get_only_dones(arrange_data(lines), start, end)

    plot_all(data_dict, save_folder, last_epochs, show, start)


def plot_all_from_folder(folder_path: str, save_folder: str, last_epochs: int = 100, show=False,
                         start=0, end=-1):
    lines = csv_handler.load_all_data(os.path.join(folder_path, CSV_FILE))
    plot_all_from_lines(lines, save_folder, last_epochs, show, start, end)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GENETIC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_all_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str, show=False):
    plot_avg_distance_per_generation(data_dict, save_folder, show)
    plot_avg_percentage_in_target_per_generation(data_dict, save_folder, show)
    plot_avg_collision_success_per_generation(data_dict, save_folder, show)
    plot_avg_max_total_reward_generation(data_dict, save_folder, show)


def plot_all_generation_from_lines(lines: List[List[Any]], save_folder: str, show=False,
                                   start=0, end=-1):
    data_dict = get_only_dones(arrange_data(lines), start, end)
    plot_all_generation(data_dict, save_folder, show)


def plot_all_generation_from_folder(folder_path: str, save_folder: str, show=False,
                                    start=0, end=-1):
    lines = csv_handler.load_all_data(os.path.join(folder_path, CSV_FILE))
    plot_all_generation_from_lines(lines, save_folder, show, start, end)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPARE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_all_compare_from_folders(folders: List[str], save_folder: str, last_epochs: int = 100, show=False,
                                  start=0, end=-1):
    names = []
    dicts = []
    for folder in folders:
        lines = csv_handler.load_all_data(os.path.join(folder, CSV_FILE))
        data_dict = get_only_dones(arrange_data(lines), start, end)
        names.append(folder)
        dicts.append(data_dict)
    plot_avg_distance_compare(dicts, names, save_folder, show, start)
    plot_avg_epoch_distance_compare(dicts, names, save_folder, last_epochs, show, start)
    plot_avg_percentage_compare(dicts, names, save_folder, show, start)
    plot_avg_epoch_percentage_compare(dicts, names, save_folder, last_epochs, show, start)
    plot_avg_success_compare(dicts, names, save_folder, show, start)
    plot_avg_epoch_success_compare(dicts, names, save_folder, last_epochs, show, start)
    plot_avg_collision_compare(dicts, names, save_folder, show, start)
    plot_avg_epoch_collision_compare(dicts, names, save_folder, last_epochs, show, start)

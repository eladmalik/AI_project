import os.path
from typing import List, Any, Dict

import matplotlib
from matplotlib import pyplot as plt
from distutils.util import strtobool
from utils.enums import DataType

data_converters = {
    DataType.LAST_REWARD: float,
    DataType.TOTAL_REWARD: float,
    DataType.DISTANCE_TO_TARGET: float,
    DataType.PERCENTAGE_IN_TARGET: float,
    DataType.ANGLE_TO_TARGET: float,
    DataType.SUCCESS: strtobool,
    DataType.COLLISION: strtobool
}


def arrange_data(lines: List[List[Any]]):
    data_dict = dict()
    for j in range(len(lines[0])):
        data_type = DataType(int(lines[0][j]))
        data_dict[data_type] = [data_converters[data_type](lines[i][j]) for i in range(1, len(lines))]
    return data_dict


def plot_distance_data(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    distance_color = "#1a1fab"
    avg_distance_color = "#e44dff"
    last_epochs_color = "#2af76b"

    distances = data_dict[DataType.DISTANCE_TO_TARGET]
    avg_distances = [sum(distances[:i]) / i for i in range(1, len(distances))]

    last_epochs_indx = list(range(last_epochs - 1, len(distances)))
    avg_last_epochs = [sum(distances[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                       last_epochs_indx]
    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Distance To Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Distance to Target', color=axes_color)

    plt.plot(distances, label="distance", color=distance_color, marker='o')
    plt.plot(avg_distances, label="mean distance", color=avg_distance_color)
    plt.plot(last_epochs_indx, avg_last_epochs, label=f"last {last_epochs} simulations mean",
             color=last_epochs_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "distances.png"))


def plot_percentage_in_target_data(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    percentage_color = "#1a1fab"
    avg_percentage_color = "#e44dff"
    last_epochs_color = "#2af76b"

    in_targert_percentage = data_dict[DataType.PERCENTAGE_IN_TARGET]
    avg = [sum(in_targert_percentage[:i]) / i for i in range(1, len(in_targert_percentage))]

    last_epochs_indx = list(range(last_epochs - 1, len(in_targert_percentage)))
    avg_last_epochs = [sum(in_targert_percentage[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                       in last_epochs_indx]
    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Agent Containment In Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('% of containment in Target', color=axes_color)

    plt.plot(in_targert_percentage, label="% in target", color=percentage_color, marker='o')
    plt.plot(avg, label="mean % in target", color=avg_percentage_color)
    plt.plot(last_epochs_indx, avg_last_epochs, label=f"last {last_epochs} simulations mean",
             color=last_epochs_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "percetage_in_target.png"))


def plot_success_collision_rate(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    avg_success_color = "#3700de"
    last_epochs_success_color = "#00def2"

    avg_collision_color = "#bf0000"
    last_epochs_collision_color = "#d95300"

    success = data_dict[DataType.SUCCESS]
    avg_success = [sum(success[:i]) / i for i in range(1, len(success))]

    last_epochs_indx = list(range(last_epochs - 1, len(success)))
    avg_success_last_epochs = [sum(success[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                               in last_epochs_indx]

    collision = data_dict[DataType.COLLISION]
    avg_collision = [sum(collision[:i]) / i for i in range(1, len(collision))]
    avg_collision_last_epochs = [sum(collision[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                                 in last_epochs_indx]

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Agent Success and Collision Rate', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Success Rate', color=axes_color)

    plt.plot(avg_success, label="mean success rate", color=avg_success_color)
    plt.plot(last_epochs_indx, avg_success_last_epochs, label=f"last {last_epochs} simulations success mean",
             color=last_epochs_success_color)

    plt.plot(avg_collision, label="mean collision rate", color=avg_collision_color)
    plt.plot(last_epochs_indx, avg_collision_last_epochs, label=f"last {last_epochs} simulations collision "
                                                                f"mean",
             color=last_epochs_collision_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "success_and_collision.png"))


def plot_all(data_dict: Dict[DataType, List[Any]], save_folder: str, last_epochs: int = 100, show=False):
    plot_distance_data(data_dict, save_folder, last_epochs, show)
    plot_percentage_in_target_data(data_dict, save_folder, last_epochs, show)
    plot_success_collision_rate(data_dict, save_folder, last_epochs, show)


def plot_all_from_lines(lines: List[List[Any]], save_folder: str, last_epochs: int = 100, show=False):
    data_dict = arrange_data(lines)
    plot_all(data_dict, save_folder, last_epochs, show)

# if __name__ == '__main__':
#     lines = csv_handler.load_all_data(
#         os.path.join("..", "model", "PPO_LSTM_26-08-2022__19-51-28", "results.csv"))
#     data = arrange_data(lines)
#     plot_distance_data(data, "tmp", last_epochs=100)
#     plot_percentage_in_target_data(data, "tmp", last_epochs=100)
#     plot_success_collision_rate(data, "tmp", last_epochs=100)

import os

import matplotlib
from matplotlib import pyplot as plt

from utils.enums import StatsType


def plot_distance_data(data_dict, save_folder, last_epochs=100, show=False, start=0):
    bg_color = "#191919"
    axes_color = 'white'
    distance_color = "#1a1fab"
    avg_distance_color = "#e44dff"
    last_epochs_color = "#2af76b"

    distances = data_dict[StatsType.DISTANCE_TO_TARGET]
    avg_distances = [sum(distances[:i]) / i for i in range(1, len(distances) + 1)]
    indexes = list(range(start, start + len(distances)))

    last_epochs_indx = list(range(last_epochs - 1, len(distances)))
    avg_last_epochs = [sum(distances[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                       last_epochs_indx]
    last_epochs_indx = list(range(start + last_epochs - 1, start + len(distances)))

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

    plt.plot(indexes, distances, label="distance", color=distance_color, marker='o')
    plt.plot(indexes, avg_distances, label="mean distance", color=avg_distance_color)
    plt.plot(last_epochs_indx, avg_last_epochs, label=f"last {last_epochs} simulations mean",
             color=last_epochs_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "distances.png"))
    plt.close()


def plot_percentage_in_target_data(data_dict, save_folder, last_epochs=100, show=False, start=0):
    bg_color = "#191919"
    axes_color = 'white'
    percentage_color = "#1a1fab"
    avg_percentage_color = "#e44dff"
    last_epochs_color = "#2af76b"

    in_targert_percentage = data_dict[StatsType.PERCENTAGE_IN_TARGET]
    avg = [sum(in_targert_percentage[:i]) / i for i in range(1, len(in_targert_percentage) + 1)]
    indexes = list(range(start, start + len(in_targert_percentage)))

    last_epochs_indx = list(range(last_epochs - 1, len(in_targert_percentage)))
    avg_last_epochs = [sum(in_targert_percentage[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                       in last_epochs_indx]
    last_epochs_indx = list(range(start + last_epochs - 1, start + len(in_targert_percentage)))
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

    plt.plot(indexes, in_targert_percentage, label="% in target", color=percentage_color, marker='o')
    plt.plot(indexes, avg, label="mean % in target", color=avg_percentage_color)
    plt.plot(last_epochs_indx, avg_last_epochs, label=f"last {last_epochs} simulations mean",
             color=last_epochs_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "percentage_in_target.png"))
    plt.close()


def plot_success_collision_rate(data_dict, save_folder, last_epochs=100, show=False, start=0):
    bg_color = "#191919"
    axes_color = 'white'
    avg_success_color = "#3700de"
    last_epochs_success_color = "#00def2"

    avg_collision_color = "#bf0000"
    last_epochs_collision_color = "#d95300"

    success = data_dict[StatsType.SUCCESS]
    avg_success = [sum(success[:i]) / i for i in range(1, len(success) + 1)]

    last_epochs_indx = list(range(last_epochs - 1, len(success)))
    avg_success_last_epochs = [sum(success[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                               in last_epochs_indx]

    collision = data_dict[StatsType.COLLISION]
    avg_collision = [sum(collision[:i]) / i for i in range(1, len(collision) + 1)]
    avg_collision_last_epochs = [sum(collision[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                                 in last_epochs_indx]
    indexes = list(range(start, start + len(success)))
    last_epochs_indx = list(range(start + last_epochs - 1, start + len(success)))

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

    plt.plot(indexes, avg_success, label="mean success rate", color=avg_success_color)
    plt.plot(last_epochs_indx, avg_success_last_epochs, label=f"last {last_epochs} simulations success mean",
             color=last_epochs_success_color)

    plt.plot(indexes, avg_collision, label="mean collision rate", color=avg_collision_color)
    plt.plot(last_epochs_indx, avg_collision_last_epochs, label=f"last {last_epochs} simulations collision "
                                                                f"mean",
             color=last_epochs_collision_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "success_and_collision.png"))
    plt.close()

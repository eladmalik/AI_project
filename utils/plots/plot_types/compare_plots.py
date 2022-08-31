import os
from typing import List, Dict, Any

import matplotlib
from matplotlib import pyplot as plt

from utils.enums import StatsType


def plot_avg_distance_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                              show=False,
                              start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        distances = data_dict[StatsType.DISTANCE_TO_TARGET]
        avg_distances = [sum(distances[:i]) / i for i in range(1, len(distances) + 1)]
        indexes = list(range(start, start + len(distances)))
        plot_data.append((indexes, avg_distances))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Average Distance To Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Distance to Target', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "avg_distances.png"))
    plt.close()


def plot_avg_epoch_distance_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                                    last_epochs=100,
                                    show=False,
                                    start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        distances = data_dict[StatsType.DISTANCE_TO_TARGET]
        last_epochs_indx = list(range(last_epochs - 1, len(distances)))
        avg_last_epochs = [sum(distances[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                           last_epochs_indx]
        last_epochs_indx = list(range(start + last_epochs - 1, start + len(distances)))
        plot_data.append((last_epochs_indx, avg_last_epochs))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title(f'Average of {last_epochs} Last Epochs Distance To Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Distance to Target', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, f"avg_last_{last_epochs}_epochs_distances.png"))
    plt.close()


def plot_avg_percentage_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                                show=False,
                                start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        percentage = data_dict[StatsType.PERCENTAGE_IN_TARGET]
        avg_percentage = [sum(percentage[:i]) / i for i in range(1, len(percentage) + 1)]
        indexes = list(range(start, start + len(percentage)))
        plot_data.append((indexes, avg_percentage))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Average Agent Containment In Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Containment In Target', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "avg_containment_in_target.png"))
    plt.close()


def plot_avg_epoch_percentage_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                                      last_epochs=100,
                                      show=False,
                                      start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        percentage = data_dict[StatsType.PERCENTAGE_IN_TARGET]
        last_epochs_indx = list(range(last_epochs - 1, len(percentage)))
        avg_last_epochs = [sum(percentage[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                           last_epochs_indx]
        last_epochs_indx = list(range(start + last_epochs - 1, start + len(percentage)))
        plot_data.append((last_epochs_indx, avg_last_epochs))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title(f'Average of {last_epochs} Last Epochs Containment In Target Parking', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Containment In Target', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, f"avg_last_{last_epochs}_epochs_containment.png"))
    plt.close()


def plot_avg_success_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                             show=False,
                             start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        success = data_dict[StatsType.SUCCESS]
        avg_success = [sum(success[:i]) / i for i in range(1, len(success) + 1)]
        indexes = list(range(start, start + len(success)))
        plot_data.append((indexes, avg_success))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Average Success Rate', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Success Rate', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "avg_success.png"))
    plt.close()


def plot_avg_epoch_success_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                                   last_epochs=100,
                                   show=False,
                                   start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        success = data_dict[StatsType.SUCCESS]
        last_epochs_indx = list(range(last_epochs - 1, len(success)))
        avg_last_epochs = [sum(success[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                           last_epochs_indx]
        last_epochs_indx = list(range(start + last_epochs - 1, start + len(success)))
        plot_data.append((last_epochs_indx, avg_last_epochs))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title(f'Average of {last_epochs} Last Epochs Success Rate', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Success Rate', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, f"avg_last_{last_epochs}_epochs_success.png"))
    plt.close()


def plot_avg_collision_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                               show=False,
                               start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        collision = data_dict[StatsType.COLLISION]
        avg_collision = [sum(collision[:i]) / i for i in range(1, len(collision) + 1)]
        indexes = list(range(start, start + len(collision)))
        plot_data.append((indexes, avg_collision))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Average Collision Rate', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Collision Rate', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "avg_collision.png"))
    plt.close()


def plot_avg_epoch_collision_compare(data_dicts: List[Dict[StatsType, Any]], names: List[str], save_folder,
                                     last_epochs=100,
                                     show=False,
                                     start=0):
    bg_color = "#191919"
    axes_color = 'white'
    plot_data = list()
    for i in range(len(names)):
        data_dict = data_dicts[i]
        collision = data_dict[StatsType.COLLISION]
        last_epochs_indx = list(range(last_epochs - 1, len(collision)))
        avg_last_epochs = [sum(collision[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch in
                           last_epochs_indx]
        last_epochs_indx = list(range(start + last_epochs - 1, start + len(collision)))
        plot_data.append((last_epochs_indx, avg_last_epochs))

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title(f'Average of {last_epochs} Last Epochs Collision Rate', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Collision Rate', color=axes_color)
    for i in range(len(names)):
        plt.plot(plot_data[i][0], plot_data[i][1], label=names[i])
    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, f"avg_last_{last_epochs}_epochs_collision.png"))
    plt.close()

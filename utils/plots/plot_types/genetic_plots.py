import os
from typing import Dict, List, Any

import matplotlib
from matplotlib import pyplot as plt

from utils.enums import StatsType


def plot_avg_distance_per_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str, show=False):
    generations = dict()
    for i in range(len(data_dict[StatsType.GENERATION])):
        generation = data_dict[StatsType.GENERATION][i]
        if generation not in generations:
            generations[generation] = [0, 0]
        generations[generation][0] += 1
        generations[generation][1] += data_dict[StatsType.DISTANCE_TO_TARGET][i]
    for generation in generations:
        generations[generation] = generations[generation][1] / generations[generation][0]

    distances_items = list(generations.items())
    distances_items.sort(key=lambda x: x[0])
    indexes = [item[0] for item in distances_items]
    distances = [item[1] for item in distances_items]

    bg_color = "#191919"
    axes_color = 'white'
    distance_color = "#e44dff"

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
    plt.xlabel('Number of Generation', color=axes_color)
    plt.ylabel('Distance to Target', color=axes_color)

    plt.plot(indexes, distances, label="distance", color=distance_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "distances.png"))
    plt.close()


def plot_avg_percentage_in_target_per_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str,
                                                 show=False):

    generations = dict()
    for i in range(len(data_dict[StatsType.GENERATION])):
        generation = data_dict[StatsType.GENERATION][i]
        if generation not in generations:
            generations[generation] = [0, 0]
        generations[generation][0] += 1
        generations[generation][1] += data_dict[StatsType.PERCENTAGE_IN_TARGET][i]
    for generation in generations:
        generations[generation] = generations[generation][1] / generations[generation][0]

    percentage_in_target_items = list(generations.items())
    percentage_in_target_items.sort(key=lambda x: x[0])
    indexes = [item[0] for item in percentage_in_target_items]
    p_in_target = [item[1] for item in percentage_in_target_items]

    bg_color = "#191919"
    axes_color = 'white'
    percentage_color = "#e44dff"

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Average Containment Percentage In Target Parking', color=axes_color)
    plt.xlabel('Number of Generation', color=axes_color)
    plt.ylabel('Containment in target parking', color=axes_color)

    plt.plot(indexes, p_in_target, label="Containment in target", color=percentage_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "percentage_in_target.png.png"))
    plt.close()


def plot_avg_collision_success_per_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str,
                                              show=False):
    generations = dict()
    for i in range(len(data_dict[StatsType.GENERATION])):
        generation = data_dict[StatsType.GENERATION][i]
        if generation not in generations:
            generations[generation] = [0, 0, 0]
        generations[generation][0] += 1
        generations[generation][1] += data_dict[StatsType.COLLISION][i]
        generations[generation][2] += data_dict[StatsType.SUCCESS][i]
    for generation in generations:
        generations[generation] = (generations[generation][1] / generations[generation][0],
                                   generations[generation][2] / generations[generation][0])

    percentage_in_target_items = list(generations.items())
    percentage_in_target_items.sort(key=lambda x: x[0])
    indexes = [item[0] for item in percentage_in_target_items]
    collisions = [item[1][0] for item in percentage_in_target_items]
    success = [item[1][1] for item in percentage_in_target_items]

    bg_color = "#191919"
    axes_color = 'white'
    avg_success_color = "#3700de"
    avg_collision_color = "#bf0000"

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
    plt.xlabel('Number of Generation', color=axes_color)
    plt.ylabel('Success Rate', color=axes_color)

    plt.plot(indexes, success, label="mean success rate", color=avg_success_color)

    plt.plot(indexes, collisions, label="mean collision rate", color=avg_collision_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "success_and_collision.png"))
    plt.close()


def plot_avg_max_total_reward_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str,
                                         show=False):

    gen_max = dict()
    generations = dict()
    for i in range(len(data_dict[StatsType.GENERATION])):
        generation = data_dict[StatsType.GENERATION][i]
        if generation not in generations:
            gen_max[generation] = 0
            generations[generation] = [0, 0]
        generations[generation][0] += 1
        generations[generation][1] += data_dict[StatsType.TOTAL_REWARD][i]
        if gen_max[generation] < data_dict[StatsType.TOTAL_REWARD][i]:
            gen_max[generation] = data_dict[StatsType.TOTAL_REWARD][i]
    for generation in generations:
        generations[generation] = generations[generation][1] / generations[generation][0]

    total_reward_avg_items = list(generations.items())
    total_reward_avg_items.sort(key=lambda x: x[0])

    total_reward_max_items = list(gen_max.items())
    total_reward_max_items.sort(key=lambda x: x[0])

    indexes = [item[0] for item in total_reward_avg_items]
    avg_total_scores = [item[1] for item in total_reward_avg_items]
    max_total_scores = [item[1] for item in total_reward_max_items]

    bg_color = "#191919"
    axes_color = 'white'
    avg_reward_color = "#e44dff"
    max_reward_color = "#2af76b"

    plt.clf()
    plt.figure(facecolor=bg_color)
    ax = plt.axes()
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('white')
    ax.set_facecolor(color=bg_color)
    plt.title('Max and Average Total Rewards', color=axes_color)
    plt.xlabel('Number of Simulations', color=axes_color)
    plt.ylabel('Max and Average Total Rewards', color=axes_color)

    plt.plot(indexes, avg_total_scores, label="avg total reward", color=avg_reward_color)
    plt.plot(indexes, max_total_scores, label="max total reward", color=max_reward_color)

    plt.ylim(ymin=0)
    plt.legend()
    fig = plt.gcf()
    if show:
        plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "total_reward.png"))
    plt.close()
import os.path
from typing import List, Any, Dict

import matplotlib
from matplotlib import pyplot as plt
from distutils.util import strtobool
from utils.enums import StatsType

data_converters = {
    StatsType.LAST_REWARD: float,
    StatsType.TOTAL_REWARD: float,
    StatsType.DISTANCE_TO_TARGET: float,
    StatsType.PERCENTAGE_IN_TARGET: float,
    StatsType.ANGLE_TO_TARGET: float,
    StatsType.SUCCESS: strtobool,
    StatsType.COLLISION: strtobool,
    StatsType.GENERATION: int,
    StatsType.I_EPISODE: int,
    StatsType.I_STEP: int,
    StatsType.IS_DONE: strtobool
}


def arrange_data(lines: List[List[Any]]):
    data_dict = dict()
    for j in range(len(lines[0])):
        data_type = StatsType(int(lines[0][j]))
        data_dict[data_type] = [data_converters[data_type](lines[i][j]) for i in range(1, len(lines))]
    return data_dict


def get_only_dones(data_dict):
    new_dict = {key: list() for key in data_dict}

    for i in range(len(data_dict[StatsType.IS_DONE])):
        if data_dict[StatsType.IS_DONE][i]:
            for key in data_dict:
                new_dict[key].append(data_dict[key][i])
    return new_dict


def plot_distance_data(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    distance_color = "#1a1fab"
    avg_distance_color = "#e44dff"
    last_epochs_color = "#2af76b"

    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)

    distances = data_dict[StatsType.DISTANCE_TO_TARGET]
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
    plt.close()


def plot_percentage_in_target_data(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    percentage_color = "#1a1fab"
    avg_percentage_color = "#e44dff"
    last_epochs_color = "#2af76b"

    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)

    in_targert_percentage = data_dict[StatsType.PERCENTAGE_IN_TARGET]
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
    fig.savefig(os.path.join(save_folder, "percentage_in_target.png"))
    plt.close()


def plot_success_collision_rate(data_dict, save_folder, last_epochs=100, show=False):
    bg_color = "#191919"
    axes_color = 'white'
    avg_success_color = "#3700de"
    last_epochs_success_color = "#00def2"

    avg_collision_color = "#bf0000"
    last_epochs_collision_color = "#d95300"

    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)

    success = data_dict[StatsType.SUCCESS]
    avg_success = [sum(success[:i]) / i for i in range(1, len(success))]

    last_epochs_indx = list(range(last_epochs - 1, len(success)))
    avg_success_last_epochs = [sum(success[epoch - last_epochs + 1:epoch + 1]) / last_epochs for epoch
                               in last_epochs_indx]

    collision = data_dict[StatsType.COLLISION]
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
    plt.close()


def plot_all(data_dict: Dict[StatsType, List[Any]], save_folder: str, last_epochs: int = 100, show=False):
    plot_distance_data(data_dict, save_folder, last_epochs, show)
    plot_percentage_in_target_data(data_dict, save_folder, last_epochs, show)
    plot_success_collision_rate(data_dict, save_folder, last_epochs, show)


def plot_all_from_lines(lines: List[List[Any]], save_folder: str, last_epochs: int = 100, show=False):
    data_dict = arrange_data(lines)
    plot_all(data_dict, save_folder, last_epochs, show)


def plot_avg_distance_per_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str, show=False):
    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)
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
    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)

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
    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)
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
    if StatsType.IS_DONE in data_dict:
        data_dict = get_only_dones(data_dict)

    gen_max = dict()
    generations = dict()
    for i in range(len(data_dict[StatsType.GENERATION])):
        generation = data_dict[StatsType.GENERATION][i]
        if generation not in generations:
            gen_max[i] = 0
            generations[generation] = [0, 0]
        generations[generation][0] += 1
        generations[generation][1] += data_dict[StatsType.TOTAL_REWARD][i]
        if gen_max[i] < data_dict[StatsType.TOTAL_REWARD][i]:
            gen_max = data_dict[StatsType.TOTAL_REWARD][i]
    for generation in generations:
        generations[generation] = generations[generation][1] / generations[generation][0]

    total_reward_avg_items = list(generations.items())
    total_reward_avg_items.sort(key=lambda x: x[0])

    total_reward_max_items = list(gen_max.items())
    total_reward_max_items.sort(key=lambda x: x[0])
    indexes = [item[0] for item in total_reward_avg_items]
    avg_total_scores = [item[1][0] for item in total_reward_avg_items]
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


def plot_all_generation(data_dict: Dict[StatsType, List[Any]], save_folder: str, show=False):
    plot_avg_distance_per_generation(data_dict, save_folder, show)
    plot_avg_percentage_in_target_per_generation(data_dict, save_folder, show)
    plot_avg_collision_success_per_generation(data_dict, save_folder, show)


def plot_all_generation_from_lines(lines: List[List[Any]], save_folder: str, show=False):
    data_dict = arrange_data(lines)
    plot_all_generation(data_dict, save_folder, show)

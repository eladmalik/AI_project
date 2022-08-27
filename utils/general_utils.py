import functools
import json
import os.path
from datetime import datetime
from typing import Dict, Any, Callable

from simulation.CarSimSprite import CarSimSprite
import matplotlib.pyplot as plt
from IPython import display

from utils.enums import Movement, Steering

action_mapping = {
    0: (Movement.NEUTRAL, Steering.NEUTRAL),
    1: (Movement.NEUTRAL, Steering.LEFT),
    2: (Movement.NEUTRAL, Steering.RIGHT),
    3: (Movement.FORWARD, Steering.NEUTRAL),
    4: (Movement.FORWARD, Steering.LEFT),
    5: (Movement.FORWARD, Steering.RIGHT),
    6: (Movement.BACKWARD, Steering.NEUTRAL),
    7: (Movement.BACKWARD, Steering.LEFT),
    8: (Movement.BACKWARD, Steering.RIGHT),
    9: (Movement.BRAKE, Steering.NEUTRAL),
    10: (Movement.BRAKE, Steering.LEFT),
    11: (Movement.BRAKE, Steering.RIGHT)
}


def mask_subset_percentage(big_sprite: CarSimSprite, small_sprite: CarSimSprite):
    """
    This function checks how much of the small sprite's mask is inside the big sprite's mask.
    :param big_sprite: The containing sprite
    :param small_sprite: The contained sprite
    :return: a percentage (float between 0 and 1) of how much of the small sprite overlaps with the big sprite
    """
    bits_small_mask = small_sprite.mask.count()  # the amount of pixels which the mask holds
    offset = (big_sprite.rect.x - small_sprite.rect.x), (big_sprite.rect.y - small_sprite.rect.y)
    return small_sprite.mask.overlap_area(big_sprite.mask, offset) / bits_small_mask


def get_time():
    """
    :return: the current time
    """
    now = datetime.now()
    return now.strftime("%d-%m-%Y__%H-%M-%S")


def get_agent_output_folder(agent_type: str) -> str:
    folder = os.path.join("model", f'{agent_type}_{get_time()}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def dump_to_json(info_dict: Dict[str, Any], folder: str, filename: str):
    with open(os.path.join(folder, filename), "w") as file:
        json.dump(info_dict, file, indent=4)


def dump_arguments(agent_type: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            save_folder = get_agent_output_folder(agent_type)
            dump_to_json({key: str(kwargs[key]) for key in kwargs}, save_folder, "arguments.json")
            kwargs["save_folder"] = save_folder
            output = func(*args, **kwargs)
            return output

        return wrapper

    return decorator


def plot_distances(distances, mean_distances, save_folder):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Distance to Target')
    plt.plot(distances)
    plt.plot(mean_distances)
    plt.ylim(ymin=0)
    plt.text(len(distances) - 1, distances[-1], "distance")
    plt.text(len(mean_distances) - 1, mean_distances[-1], "avg distance")
    fig = plt.gcf()
    plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "distances.png"))
    plt.pause(.1)


def plot_rewards(rewards, mean_rewards, save_folder):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.plot(mean_rewards)
    plt.ylim(ymin=0)
    plt.text(len(rewards) - 1, rewards[-1], "reward")
    plt.text(len(mean_rewards) - 1, mean_rewards[-1], "avg reward")
    fig = plt.gcf()
    plt.show(block=False)
    fig.savefig(os.path.join(save_folder, "rewards.png"))
    plt.pause(.1)

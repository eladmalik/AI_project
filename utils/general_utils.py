import functools
import json
import os.path
from datetime import datetime
from typing import Dict, Any

from simulation.CarSimSprite import CarSimSprite
from utils.csv_handler import csv_handler

from utils.enums import Movement, Steering, StatsType

ARGUMENTS_FILE = "arguments.json"

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
            dump_to_json({key: str(kwargs[key]) for key in kwargs}, save_folder, ARGUMENTS_FILE)
            kwargs["save_folder"] = save_folder
            output = func(*args, **kwargs)
            return output

        return wrapper

    return decorator


def write_stats(result_writer: csv_handler, i_episode, i_step, reward, total_reward, distance, percentage,
                angle, success,
                collision, done):
    result_writer.write_row({
        StatsType.I_EPISODE: i_episode,
        StatsType.I_STEP: i_step,
        StatsType.LAST_REWARD: reward,
        StatsType.TOTAL_REWARD: total_reward,
        StatsType.DISTANCE_TO_TARGET: distance,
        StatsType.PERCENTAGE_IN_TARGET: percentage,
        StatsType.ANGLE_TO_TARGET: angle,
        StatsType.SUCCESS: success,
        StatsType.COLLISION: collision,
        StatsType.IS_DONE: done
    })

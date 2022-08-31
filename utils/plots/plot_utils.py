from distutils.util import strtobool
from typing import List, Any

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


def arrange_data(lines: List[List[Any]], start=0, end=-1):
    data_dict = dict()
    if start < 0:
        start = 0
    if end <= -1 or end > len(lines):
        end = len(lines)
    assert start <= end
    for j in range(len(lines[0])):
        data_type = StatsType(int(lines[0][j]))
        data_dict[data_type] = [data_converters[data_type](lines[i][j]) for i in range(1, len(lines))]
    return data_dict


def get_only_dones(data_dict, start: int = 0, end: int = -1):
    if StatsType.IS_DONE in data_dict:
        temp_dict = {key: list() for key in data_dict}
        for i in range(len(data_dict[StatsType.IS_DONE])):
            if data_dict[StatsType.IS_DONE][i]:
                for key in data_dict:
                    temp_dict[key].append(data_dict[key][i])
    else:
        temp_dict = data_dict

    n = len(temp_dict[StatsType.DISTANCE_TO_TARGET])
    if start < 0:
        start = 0
    if end <= -1 or end > n:
        end = n
    assert start <= end
    new_dict = {key: list() for key in temp_dict}
    for i in range(start, end):
        for key in temp_dict:
            new_dict[key].append(temp_dict[key][i])
    return new_dict

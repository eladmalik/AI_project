import math

from utils.enums import Results


def get_angle_to_target(agent, target) -> float:
    agent_vector = agent.front - agent.location
    to_target_vector = target.location - agent.location
    angle_to_target = agent_vector.angle_to(to_target_vector)
    return angle_to_target


def get_distance_to_target(agent, target) -> float:
    return agent.location.distance_to(target.location)


def get_agent_parking_cos(agent, target, results, min_percentage_in_target=1, angle_tolerance_degrees=0) -> \
        float:
    agent_vector = agent.front - agent.location
    target_vector = target.front - target.location
    angle = agent_vector.angle_to(target_vector)
    cos_angle = abs(math.cos(math.radians(angle)))
    if results[Results.PERCENTAGE_IN_TARGET] >= min_percentage_in_target and \
            agent.velocity.magnitude() == 0 and \
            cos_angle >= math.cos(math.radians(angle_tolerance_degrees)):
        return cos_angle
    return 0


def is_agent_in_parking_slot(results):
    return results[Results.PERCENTAGE_IN_TARGET] > 0


def get_drive_direction(agent):
    drive_direction = 0
    if agent.velocity.x > 0:
        drive_direction = 1
    elif agent.velocity.x < 0:
        drive_direction = -1
    return drive_direction

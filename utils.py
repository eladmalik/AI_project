from datetime import datetime

from CarSimSprite import CarSimSprite
import matplotlib.pyplot as plt
from IPython import display

plt.ion()


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


def plot_distances(distances, mean_distances):
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
    plt.show(block=False)
    plt.pause(.1)


def plot_rewards(rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.ylim(ymin=0)
    plt.text(len(rewards) - 1, rewards[-1], "reward")
    plt.show(block=False)
    plt.pause(.1)

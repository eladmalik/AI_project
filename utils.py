import pygame

from CarSimSprite import CarSimSprite


def mask_subset_percentage_old(big_mask: pygame.mask.Mask, big_mask_rect: pygame.Rect,
                               small_mask: pygame.mask.Mask, small_mask_rect: pygame.Rect):
    """

    :param big_mask:
    :param big_mask_rect:
    :param small_mask:
    :param small_mask_rect:
    :return:
    """
    bits_small_mask = small_mask.count()
    offset = (big_mask_rect.x - small_mask_rect.x), (big_mask_rect.y - small_mask_rect.y)
    return small_mask.overlap_area(big_mask, offset) / bits_small_mask


def mask_subset_percentage(big_sprite: CarSimSprite, small_sprite: CarSimSprite):
    """

    :param big_sprite:
    :param small_sprite:
    :return:
    """
    bits_small_mask = small_sprite.mask.count()  # small_mask.count()
    offset = (big_sprite.rect.x - small_sprite.rect.x), (big_sprite.rect.y - small_sprite.rect.y)
    return small_sprite.mask.overlap_area(big_sprite.mask, offset) / bits_small_mask

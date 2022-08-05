from CarSimSprite import CarSimSprite


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

import numpy as np


def _prep_mask_images(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    return im1, im2


def compare_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters

    *Implementation from:* https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137

    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    empty_score: Score to return in the event that both mask images are empty
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1, im2 = _prep_mask_images(im1, im2)

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def compare_jaccard(im1, im2):
    """Jaccard Similarity Index

    For binary mask images, this is equivalent to accuracy (i.e. number of pixels
    in agreement divided by number of pixels)
    """
    im1, im2 = _prep_mask_images(im1, im2)
    return (im1 == im2).mean()

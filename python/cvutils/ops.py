from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
import numpy as np


def pad_around_center(img, shape, mode='constant'):
    """Pad an image to a target shape with centering

    Args:
        img: Image to pad (any number of dimensions); must have size <= `shape` in all dimensions
        shape: Target shape to pad to
        mode: Padding mode (see numpy.pad)
    """
    imgs, ts = np.array(img.shape), np.array(shape)
    if np.any(imgs > ts):
        raise ValueError(
            'Cannot pad image with shape {} to target shape {} since at least one dimension is already larger'
            .format(imgs, ts)
        )
    # Compute lower padding as half the difference in shapes (with downward rounding)
    pad_lo = (ts - imgs) // 2
    # Set upper padding as remainder
    pad_hi = ts - imgs - pad_lo
    assert np.all(pad_hi >= 0)
    # Apply padding
    return np.pad(img, list(zip(pad_lo, pad_hi)), mode=mode)


DEFAULT_COLORS = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [0, 1, 1],  # Cyan
    [1, 0, 1],  # Magenta
    [1, 1, 1],  # Gray
]


def blend_image_channels(img, mix=None, colors=DEFAULT_COLORS):
    """Get single RGB image as blend of multiple channels

    Args:
        img: Image in CYX
        mix: Array or list-like of length C (same as number of channels) to be multiplied by
            each channel image in the blended result (the proportions/values given will all be rescaled to sum
            to 1)
        colors: Array or list-like of shape (N, 3) where each 3 item row is an rgb color channel fraction in [0, 1].
            For example [1, 0, 0] would associated all values for a channel to the color red while [1, 0, 1] would
            indicate magenta (R + B).  The length of this list does not necessarily need to match the number of
            channels and if that there are not enough colors for each channel, they will be cycled indefinitely
    """
    if img.ndim != 3:
        raise ValueError('Expecting  3 dimensions in image (image shape given = {})'.format(img.shape))

    nch = img.shape[0]
    ncolor = len(colors)

    # Default mixture proportions to ones
    if mix is None:
        mix = [1] * nch

    # Expect that there is a proportion for each channel
    if nch != len(mix):
        raise ValueError(
            'Number of mixture proportions must equal number of channels '
            '(image shape given = {}, mixture proportions = {})'
            .format(img.shape, mix)
        )

    # Rescale proportions to add to 1
    mix = np.array(mix)
    mix = mix / mix.sum()

    nr, nc = img.shape[1], img.shape[2]
    res = np.zeros((nr, nc, 3), dtype=np.float32)
    for i in range(nch):
        # Fetch channel 2D image and reshape to YX3
        rgb = np.repeat(img[i][..., np.newaxis], repeats=3, axis=-1)
        color = colors[i % ncolor]
        if len(color) != 3:
            raise ValueError('Colors given should have size 3 in second dimension; colors given = {}'.format(colors))
        res = res + rgb * np.array(color) * mix[i]
    return rescale_intensity(res, in_range='image', out_range='uint8').astype(np.uint8)

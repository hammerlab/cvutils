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


def blend_image_channels(img, mix=None, colors=None):
    """Get single RGB image by blending any number of image channels

    Args:
        img: Image in CYX or YX format (note that in YX format, this operation amounts to nothing but color conversion)
        mix: Array or list-like of length C (same as number of channels) to be multiplied by
            each channel image in the blended result
        colors: Array or list-like of shape (N, 3) where each 3 item row is an rgb color channel fraction in [0, 1].
            For example [1, 0, 0] would associated all values for a channel to the color red while [1, 0, 1] would
            indicate magenta (R + B).  The length of this list does not necessarily need to match the number of
            channels and if that there are not enough colors for each channel, they will be cycled indefinitely
    Returns:
        RGB image
    """
    if img.ndim == 2:
        img = img[np.newaxis]
    if img.ndim != 3:
        raise ValueError('Expecting  3 dimensions in image (image shape given = {})'.format(img.shape))

    colors = DEFAULT_COLORS if colors is None else colors
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
    mix = np.array(mix)

    nr, nc = img.shape[1], img.shape[2]
    res = np.zeros((nr, nc, 3), dtype=np.float32)
    for i in range(nch):
        # Fetch channel 2D image and reshape to YX3
        rgb = np.repeat(img[i][..., np.newaxis], repeats=3, axis=-1)
        color = colors[i % ncolor]
        if len(color) != 3:
            raise ValueError('Colors given should have size 3 in second dimension; colors given = {}'.format(colors))
        res = res + rgb * np.array(color) * mix[i]
    res = rescale_intensity(res.astype(img.dtype), in_range='dtype', out_range=np.uint8).astype(np.uint8)
    return res


def constrain_image_channels(img, dtype=None, ranges=None):
    """Constrain image channels to particular ranges

    Args:
        img: Image in CYX format
        dtype: Resulting image datatype; defaults to image dtype
        ranges: Array with shape (C, 2) or (1, 2) where second dimension denotes range lower and upper clipping values;
            A "None" lower or upper value indicates image minimum and maximum respectively; defaults to (None, None)
            for each channel
    Returns:
        Image with same shape as input and all pixel values for channels clipped to the corresponding range
    """
    if img.ndim == 2:
        img = img[np.newaxis]
    if img.ndim != 3:
        raise ValueError('Expecting  3 dimensions in image (image shape given = {})'.format(img.shape))

    if dtype is None:
        dtype = img.dtype

    if ranges is None:
        ranges = [[None, None]] * img.shape[0]
    ranges = np.array(ranges)
    if ranges.ndim == 1:
        # Stack ranges to (C, -1) -- not sure how many items are in second axis but that is checked next
        ranges = np.repeat(ranges[np.newaxis], img.shape[0], 0)

    # Validate that ranges have length equal to num image channels and 2 values for range per channel
    if ranges.shape[1] != 2:
        raise ValueError('Ranges must have length 2 in second dimension (ranges shape = {})'.format(ranges.shape))
    if ranges.shape[0] != img.shape[0]:
        raise ValueError(
            'Ranges must have length equal to number of image channels '
            '(ranges shape = {}, image shape = {}'.format(ranges.shape, img.shape)
        )

    # Apply clip to each channel and restack
    return np.stack([
        rescale_intensity(
            img[i], in_range=(
                img[i].min() if ranges[i, 0] is None else ranges[i, 0],
                img[i].max() if ranges[i, 1] is None else ranges[i, 1]
            ), out_range=dtype
        ).astype(dtype)
        for i in range(img.shape[0])
    ])
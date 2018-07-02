
def get_next_highest_num_divisible_by_two(n, p=6):
    """Get next highest number than n divisible by two at least `p` times

    This is helpful to avoid errors from Mask-RCNN library like:
        > Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.
        > For example, use 256, 320, 384, 448, 512, ... etc.
    """
    d = 2 ** p
    while True:
        if n % d == 0:
            return n
        n += 1

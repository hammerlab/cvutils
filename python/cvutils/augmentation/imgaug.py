import numpy as np


def apply_augmentation(image, mask, augmentation):
    """Apply imgaug augmentation pipeline to both a mask and an image

    Reference:
        - https://github.com/aleju/imgaug/issues/41
        - https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/mrcnn/model.py
    """
    import imgaug

    # Augmentors that are safe to apply to masks
    # Some, such as Affine, have settings that make them unsafe
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]

    def hook(images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

    if image.dtype != np.uint8:
        raise ValueError('Image must be of type uint8 for augmentation (given {})'.format(image.dtype))
    if mask.dtype != np.bool:
        raise ValueError('Mask must be of type boolean for augmentation (given {})'.format(mask.dtype))

    # Store shapes before augmentation to compare
    image_shape = image.shape
    mask_shape = mask.shape

    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)

    # Change mask to np.uint8 because imgaug doesn't support np.bool
    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))

    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    # Change mask back to bool
    mask = mask.astype(np.bool)

    return image, mask


class InMemoryDataLoader(object):

    def __init__(self, X, Y, processor=None, augmentation=None):
        assert len(X) == len(Y)
        self.ids = np.arange(len(X))
        self.X = X
        self.Y = Y
        self.processor = processor
        self.augmentation = augmentation
        self.enable_augmentation = True

    def set_augmentation_enabled(self, value):
        self.enable_augmentation = value

    def load(self, sample_id):
        Xi, Yi = self.X[sample_id], self.Y[sample_id]

        # Run augmentation, if provided and not disabled
        if self.augmentation is not None and self.enable_augmentation:
            Xi, Yi = apply_augmentation(Xi, Yi, self.augmentation)

        # Run post-processing (often scaling from uint8 to float)
        if self.processor is not None:
            Xi, Yi = self.processor(sample_id, Xi, Yi)

        return Xi, Yi

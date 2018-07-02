from collections import namedtuple
from mrcnn import model as mrcnn_model
import numpy as np


Prediction = namedtuple('Prediction', [
    'image', 'image_id', 'image_info',
    'pred_class_ids', 'pred_class_names', 'pred_masks', 'pred_rois', 'pred_scores',
    'true_class_ids', 'true_class_names', 'true_masks'
])


def prediction_generator(model, dataset, augmentation=None, image_ids=None, config=None):
    """Generate predictions paired with ground truth information

    Args:
        model: Mask RCNN Model instance
        dataset: Mask RCNN Dataset
        augmentation: Optional imgaug augmentation object to be applied to each ground truth image as it is loaded;
            when provided, augmentation will be applied to ground-truth images and masks and predictions will
            be provided based on augmented version as well
        image_ids: List of image ids within dataset to process; For example:
            - To process all images, leave as None
            - To process the same image with different augmentations, specify the same image_id N times
                and set augmentation parameter
            - To process a pre-selected subset, specify only those image ids
        config: Mask RCNN configuration; if not provided the configuration already associated with the model
            instance will be used
    Returns:
         A generator of Prediction instances (see inference.Prediction)
    """

    if config is None:
        config = model.config
    if config.BATCH_SIZE != 1:
        raise ValueError('Configuration batch size must = 1 for this inference method')

    if image_ids is None:
        image_ids = dataset.image_ids

    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            mrcnn_model.load_image_gt(
                dataset, config, image_id,
                use_mini_mask=False, augmentation=augmentation
            )
        detection = model.detect([image], verbose=0)[0]

        if gt_class_id.ndim != 1:
            raise ValueError('Expecting true class ids array to have ndim == 1 (shape = {})'.format(gt_class_id.shape))
        if detection['class_ids'].ndim != 1:
            raise ValueError \
                ('Expecting pred class ids array to have ndim == 1 (shape = {})'.format(detection['class_ids'].shape))
        if gt_mask.ndim != 3:
            raise ValueError('Expecting true masks array to have ndim == 3 (shape = {})'.format(gt_mask.shape))
        if detection['masks'].ndim != 3:
            raise ValueError \
                ('Expecting pred masks array to have ndim == 3 (shape = {})'.format(detection['masks'].shape))

        yield Prediction(
            image=image, image_id=image_id, image_info=dataset.image_reference(image_id),
            pred_class_ids=detection['class_ids'],
            pred_class_names=np.array([dataset.class_names[i] for i in detection['class_ids']]),
            pred_masks=detection['masks'],
            pred_rois=detection['rois'],
            pred_scores=detection['scores'],
            true_class_ids=gt_class_id,
            true_class_names=np.array([dataset.class_names[i] for i in gt_class_id]),
            true_rois=gt_bbox,
            true_masks=gt_mask
        )

"""matterport/Mask_RCNN model loading and initialization utilities"""
from mrcnn import utils
import glob
import os
import os.path as osp
import mrcnn.model as mrcnn_model

ENV_MASK_RCNN_DIR = 'MASK_RCNN_DIR'


def get_checkpoint_path(model_dir, epoch):
    # Assume that checkpoints may be stored anywhere under the model directory tree but that each
    # checkpoint file is at least identifiable by the mask_rcnn_*_{epoch}.h5 pattern
    files = glob.iglob(osp.join(model_dir, '**', 'mask_rcnn_*_{:04d}.h5'.format(epoch)), recursive=True)
    if len(files) == 0:
        raise ValueError('Failed to find checkpoint for epoch {} (model dir = {})'.format(epoch, model_dir))
    if len(files) > 1:
        raise ValueError(
            'Found multiple files for epoch {} (model dir = {}) which should not be possible; files found = []'
            .format(epoch, model_dir, files)
        )
    return osp.join(model_dir, files[0])


def get_model(mode, config, model_dir, init_with='coco', epoch=None):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    model = mrcnn_model.MaskRCNN(mode=mode, config=config, model_dir=model_dir)
    return initialize_model(model, init_with, epoch)


def initialize_model(model, init_with, epoch=None):
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":

        # Load coco weights in Mask RCNN project root as a central location to
        # avoid multiple download locations
        path = osp.join(os.environ[ENV_MASK_RCNN_DIR], 'mask_rcnn_coco.h5')
        # Download COCO trained weights from Releases if needed
        if not osp.exists(path):
            utils.download_trained_weights(path)

        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "epoch":
        if epoch is None:
            raise ValueError('Must set epoch parameter when loading model for a specific epoch')
        model.load_weights(get_checkpoint_path(model.model_dir, epoch))
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        raise ValueError(
            'Initialization mode "{}" is not valid (should be one of "imagenet", "coco", "epoch" or "last")'
            .format(init_with)
        )

    return model


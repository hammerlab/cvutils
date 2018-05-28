"""Abstract implementations of matterport/Mask_RCNN dataset"""
import os
import fnmatch
import os.path as osp
import numpy as np
from mrcnn import utils
from cvutils.rectlabel import io as rectlabel_io


class RectLabelDataset(utils.Dataset):
    """RectLabel dataset model used to encapsulate mask/image loading based on standard RectLabel dir organization"""

    @staticmethod
    def get_image_id_from_path(image_path):
        return osp.basename(image_path)

    @staticmethod
    def get_annotations(image_path):
        annot_path = rectlabel_io.get_annotation_path(image_path)
        annot_shape, annot_data = rectlabel_io.load_annotations(annot_path)
        return annot_shape, annot_data

    def initialize(self, image_paths, classes, source):
        for i, c in enumerate(classes):
            self.add_class(source, i + 1, c)
        for p in image_paths:
            self.add_image(source, image_id=RectLabelDataset.get_image_id_from_path(p), path=p)

    # Use default image loader:
    # https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/mrcnn/utils.py#L360
    # def load_image(self, image_id):
    #     pass

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID"""
        image_path = self.image_info[image_id]['path']
        shape, annotations = RectLabelDataset.get_annotations(image_path)

        # Remove any unnecessary segmentation masks
        annotations = [a for a in annotations if a.object_type in self.class_names]

        count = len(annotations)

        # If there are no annotations, return empty arrays
        # https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/mrcnn/utils.py#L373
        if count == 0:
            return super(RectLabelDataset, self).load_mask(image_id)

        # Set a single 2D mask for each annotation
        mask = np.zeros([shape[0], shape[1], count], dtype=np.uint8)
        for i, a in enumerate(annotations):
            mask[:, :, i] = a.mask

        # Determine class indexes for each mask
        class_ids = np.array([self.class_names.index(a.object_type) for a in annotations], dtype=np.int32)

        return mask.astype(np.bool), class_ids

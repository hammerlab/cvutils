"""matterport/Mask_RCNN session and GPU management utilities"""
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


def get_session(gpu_fraction=0.75):
    """Get TF session that does not pre-allocate GPU memory"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def init_keras_session(gpu_fraction=0.75):
    KTF.set_session(get_session(gpu_fraction))


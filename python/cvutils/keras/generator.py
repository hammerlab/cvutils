"""Keras data generator and augmentation utilities"""
import keras
import numpy as np


class BaseDataGenerator(keras.utils.Sequence):

    def __init__(self, ids, loader, batch_size=32, shuffle=True):
        """Compositional Keras data loader useful for injection of custom image processing logic

        This class can be used in situations where any dataset is indexable by some kind of per-case
        id (ids can be any type) and where for each case, some kind of processing logic should be applied
        to the corresponding data.

        At the very least, this is useful and necessary for injecting augmentation into a Keras generator
        where the augmentation applies to both the images and masks.

        Reference:
            - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
        Args:
            ids: A sequence of image/case ids
            loader: A callable with signature function(id) -> X, Y where `id` is a single element in `ids`
                and `X` and `Y` are the resulting independent and dependent variables, respectively
            batch_size: Batch size for generator
            shuffle: Whether or not to shuffle the ids loaded after every epoch
        """
        self.ids = ids
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    @staticmethod
    def from_arrays(X, Y, **kwargs):
        """Create a BaseDataGenerator from pre-loaded arrays"""
        if len(X) != len(Y):
            raise ValueError('X and Y should have same length; shape X = {}, shape Y = {}'.format(X.shape, Y.shape))
        ids = np.arange(len(X))

        def loader(sample_id):
            return X[sample_id], Y[sample_id]
        return BaseDataGenerator(ids, loader, **kwargs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.ids) / self.batch_size))
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]

        # Generate data
        return self.__data_generation(batch_ids)

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        # Initialization
        X = []
        Y = []

        # Generate data
        for i, sample_id in enumerate(batch_ids):
            x, y = self.loader(sample_id)
            X.append(x)
            Y.append(y)

        return np.stack(X, 0), np.stack(Y, 0)



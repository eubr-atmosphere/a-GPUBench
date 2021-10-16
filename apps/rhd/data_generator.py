import itertools

import numpy as np
import tensorflow as tf

from random import randint

class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, data_path, instances_ids, labels, batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=3, shuffle=True, crop_strategy='center'):
        self.data_path = data_path
        self.instances_ids = instances_ids
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.crop_strategy = crop_strategy

        # Generate initial instances_indexes
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.instances_indexes = np.arange(len(self.instances_ids))
        if self.shuffle == True:
            np.random.shuffle(self.instances_indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return int(np.ceil(len(self.instances_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # Generate indexes of the batch
        indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.instances_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def __data_generation(self, batch_ids):
        """
        Generates data containing batch_size samples.
        """

        # Initialization
        cur_batch_size = len(batch_ids)
        X = np.empty((cur_batch_size, *self.dim, self.n_channels))
        y = np.empty(cur_batch_size, dtype=int)

        # Generate data
        for i, instance_id in enumerate(batch_ids):
            x = np.load('{}/{}.npz'.format(self.data_path, instance_id))['frames']

            if self.crop_strategy == 'random':
                h_init = randint(0, 16)
                w_init = randint(0, 59)
                X[i, ] = x[:, h_init:h_init + 112, w_init:w_init + 112, :]  # (l, h, w, c)
            elif self.crop_strategy == 'center':
                X[i, ] = x[:, 8:120, 30:142, :]  # (l, h, w, c)
            elif self.crop_strategy == 'none':
                X[i, ] = x

            # Store correct class
            y[i] = self.labels[instance_id]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


class TwoArmedDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for the two armed C3D network.
    """

    def __init__(self, data_path, exam_ids, instances_info, batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=3, shuffle=True, crop_strategy='center', augmentation=False):
        self.data_path = data_path
        self.exam_ids = exam_ids
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.crop_strategy = crop_strategy
        self.augmentation = augmentation

        # Construct instance IDs
        self.expand_exam_ids(instances_info)

        # Generate initial instances_indexes
        self.on_epoch_end()


    def expand_exam_ids(self, instances_info):
        """
        Create instance identifiers by expanding exams into the combinations between Doppler and non-Doppler in each
        one.
        """

        if not self.augmentation:
            instances_info = instances_info[instances_info['Augmentation'].isnull()]

        self.labels = {}

        for exam_id in self.exam_ids:
            exam = instances_info[instances_info['Exam'] == exam_id]
            exam_label = exam['Diagnosis'].iloc[0]
            non_doppler = exam[exam['With_Doppler'] == False]['Sample']
            doppler = exam[exam['With_Doppler'] == True]['Sample']

            for combination in itertools.product(non_doppler, doppler):
                self.labels[combination] = exam_label

        self.instances_ids = list(self.labels.keys())

        #from IPython import embed; embed()


    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.instances_indexes = np.arange(len(self.instances_ids))
        if self.shuffle == True:
            np.random.shuffle(self.instances_indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return int(np.ceil(len(self.instances_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # Generate indexes of the batch
        indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.instances_ids[k] for k in indexes]

        return self.__data_generation(batch_ids)

    def __data_generation(self, batch_ids):
        """
        Generates data containing batch_size samples.
        """

        # Initialization
        cur_batch_size = len(batch_ids)
        X_non_doppler = np.empty((cur_batch_size, *self.dim, self.n_channels))
        X_doppler = np.empty((cur_batch_size, *self.dim, self.n_channels))
        y = np.empty(cur_batch_size, dtype=int)

        # Generate data
        for i, (non_doppler_id, doppler_id) in enumerate(batch_ids):
            x_non_doppler = np.load('{}/{}.npz'.format(self.data_path, non_doppler_id))['frames']
            x_doppler = np.load('{}/{}.npz'.format(self.data_path, doppler_id))['frames']

            if self.crop_strategy == 'random':
                h_init = randint(0, 16)
                w_init = randint(0, 59)

                X_non_doppler[i, ] = x_non_doppler[:, h_init:h_init + 112, w_init:w_init + 112, :]  # (l, h, w, c)
                X_doppler[i, ] = x_doppler[:, h_init:h_init + 112, w_init:w_init + 112, :]  # (l, h, w, c)

            elif self.crop_strategy == 'center':
                X_non_doppler[i, ] = x_non_doppler[:, 8:120, 30:142, :]  # (l, h, w, c)
                X_doppler[i, ] = x_doppler[:, 8:120, 30:142, :]  # (l, h, w, c)

            elif self.crop_strategy == 'none':
                X_non_doppler[i, ] = x_non_doppler
                X_doppler[i, ] = x_doppler

            # Store correct class
            y[i] = self.labels[(non_doppler_id, doppler_id)]

        return [X_non_doppler, X_doppler], tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

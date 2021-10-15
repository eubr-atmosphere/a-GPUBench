import time

import keras
import numpy as np


class TimeLog(keras.callbacks.Callback):

    def __init__(self, path_to_save):
        self.path_to_save = path_to_save


    def on_train_begin(self, logs=None):
        self.times = {}


    def on_train_end(self, logs=None):
        for epoch_number, batch_times in self.times.items():
            np.save('{}/{}.npy'.format(self.path_to_save, epoch_number), np.array(batch_times))


    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.times[self.current_epoch] = []


    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()


    def on_train_batch_end(self, batch, logs=None):
        self.times[self.current_epoch].append(time.time() - self.batch_start_time)

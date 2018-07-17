from __future__ import division # Bluecrystal uses python 2.7
import keras
from matplotlib import pyplot as plt
import os
from PlotLearning import PlotLearning
from SaveHistory import SaveHistory
import numpy as np
import keras.backend as K

import CaloriesDataset
from NetworkArchitectures import NetworkZhu

# Print this code on the console, useful for debugging
with open(__file__) as f: print('\n'.join(f.read().split('\n')[1:]))

Set = {'bluecrystal': True}

network_name = 'NETWORKNAME'

np.random.seed(10)

batch_size = 1024
epochs = 1000

save_dir = os.path.join(os.getcwd(),'saved_models')

(x_train, y_train, _) = CaloriesDataset.load_acc_range(TRAINSUBJ)
(x_test, y_test, _) = CaloriesDataset.load_acc_range(TESTSUBJ)
Nchannels = x_train.shape[2]

# Remove NaNs
isnan = np.isnan(y_train)
x_train = np.delete(x_train, np.where(isnan), axis=0)
y_train = np.delete(y_train, np.where(isnan), axis=0)
isnan = np.isnan(y_test)
x_test = np.delete(x_test, np.where(isnan), axis=0)
y_test = np.delete(y_test, np.where(isnan), axis=0)

# The accelerometer is stored in vectors of 1000 elements. Keep only 256
New_bufsiz = 256
x_train = x_train[:,-New_bufsiz:,:]
x_test = x_test[:,-New_bufsiz:,:]
x_train = x_train.astype('float32')/9.81;
x_test = x_test.astype('float32')/9.81;

# MC_DCNN (https://github.com/LouisFoucard/MC_DCNN/blob/master/MultiChannel_DeepConvNet.ipynb)
model = NetworkZhu(New_bufsiz, Nchannels)
model.summary()

opt = keras.optimizers.RMSprop()

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

best_model_name = '%s_best.h5' % network_name

# Callbacks
filepath = os.path.join(save_dir,best_model_name)

save_history = SaveHistory()
save_history.SetFilename(network_name,save_dir)

if Set['bluecrystal']:
    callbacks = [save_history]
else:
    plot_learning = PlotLearning()
    plot_learning.StoreTesting(x_test,y_test)
    callbacks = [plot_learning,save_history]


class MySequence(keras.utils.Sequence):
    def __init__(self, x_train, y_train, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.x_train) /
                       float(self.batch_size)).astype('int')

    def __getitem__(self, idx):
        i1 = idx * self.batch_size
        i2 = (idx+1) * self.batch_size
        batch_x = self.x_train[i1:i2, ]
        batch_y = self.y_train[i1:i2, ]

        batch_x_aug = np.copy(batch_x)
        # Randomly change the magnitude
        mag = np.random.normal(loc=1.0, scale=0.1)
        batch_x_aug *= mag

        # Randomly swap accelerometer channels
        acc1 = np.random.permutation([0, 1, 2])
        acc2 = np.random.permutation([3, 4, 5])
        batch_x_aug[..., 0:3] = batch_x_aug[..., acc1]
        batch_x_aug[..., 3:6] = batch_x_aug[..., acc2]
        
        return batch_x_aug, batch_y

my_generator = MySequence(x_train, y_train,
                          batch_size=batch_size)

model.fit_generator(my_generator,
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks, workers=16)



from __future__ import division # Bluecrystal uses python 2.7
import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from keras.callbacks import ModelCheckpoint
from PlotLearning import PlotLearning
from SaveHistory import SaveHistory
import numpy as np
import keras.backend as K

import CaloriesDataset
from NetworkArchitectures import NetworkCombined

# Print this code on the console, useful for debugging
with open(__file__) as f: print('\n'.join(f.read().split('\n')[1:]))

Set = {'bluecrystal': True,
       'buf_siz': 1000}

network_name = 'NETWORKNAME'

np.random.seed(10)

batch_size = 128
epochs = 1000

save_dir = r'/saved_models'

# Load video and accelerometer data
(xv_train, yv_train, lab_train) = CaloriesDataset.load_range_N(TRAINSUBJ, Set['buf_siz'], r'/calories_data')
(xa_train, ya_train, _) = CaloriesDataset.load_acc_range(TRAINSUBJ)
(xv_test, yv_test, lab_test) = CaloriesDataset.load_range_N(TESTSUBJ, Set['buf_siz'], r'/calories_data')
(xa_test, ya_test, _) = CaloriesDataset.load_acc_range(TESTSUBJ)

# Gravity filter
import scipy.signal
for i in range(xa_test.shape[0]):
    for c in range(6):
        xa_test[i,:,c] -= scipy.signal.wiener(xa_test[i,:,c], 30)
        
for i in range(xa_train.shape[0]):
    for c in range(6):
        xa_train[i,:,c] -= scipy.signal.wiener(xa_train[i,:,c], 30)

# Resize the accelerometers to match the video buffer size
xa_train = xa_train[:, -Set['buf_siz']:]
xa_test = xa_test[:, -Set['buf_siz']:]

# Get data size
Nv_chan = xv_train.shape[1]
img_rows = xv_train.shape[2]
img_cols = xv_train.shape[3]
Na_chan = xa_train.shape[2]
acc_buffersiz = xa_train.shape[1]

# Reshape into keras format
xv_train = np.moveaxis(xv_train, 1, 3).astype('float32')/255
xv_test = np.moveaxis(xv_test, 1, 3).astype('float32')/255
xa_train = xa_train.astype('float32')/9.81
xa_test = xa_test.astype('float32')/9.81

# Replace missing data with random sampling from same label
nan_train = np.logical_or(np.isnan(ya_train), np.isnan(yv_train))
Nlabels = 11
for li in range(0, Nlabels+1):
    print('Replacing label %d' % li)

    tbr_train = np.logical_and(lab_train == li, nan_train)
    Ntbr_tr = np.sum(tbr_train.astype('int'))
    # Randomly sample substitutes
    good_candidates_train = np.logical_and(lab_train == li,
                                           np.logical_not(nan_train))
    Ngood_train = np.sum(good_candidates_train.astype('int'))
    print('%d elements to be replaced in train. Using %d good values' %
          (Ntbr_tr, Ngood_train))

    sub_i_tr = np.random.permutation(Ngood_train)
    # If there are more NaNs than good elements, simply sample them more than
    # once
    if Ngood_train < Ntbr_tr:
        sub_i_tr = np.repeat(sub_i_tr, np.ceil(Ntbr_tr/Ngood_train))
        print('Not enough good elements for training. Data sampled %d times' %
              np.ceil(Ntbr_tr/Ngood_train))

    sub_i_tr = sub_i_tr[0:Ntbr_tr]
    good_candidates_train = np.take(np.where(good_candidates_train), sub_i_tr)
    xv_train[tbr_train, ] = np.take(xv_train, good_candidates_train, 0)
    ya_train[tbr_train] = np.take(ya_train, good_candidates_train)
    xa_train[tbr_train, ] = np.take(xa_train, good_candidates_train, 0)

# Accelerometer and silhouette training vectors are the same, they're both
# calories. They originally contain different NaN, but once these are replaced
# only one training vector is needed
y_train = ya_train

# Remove Nan data from test
nan_test = np.logical_or(np.isnan(ya_test), np.isnan(yv_test))
xv_test = np.delete(xv_test, np.where(nan_test), 0)
xa_test = np.delete(xa_test, np.where(nan_test), 0)
y_test = np.delete(yv_test, np.where(nan_test))

# Network
model = NetworkCombined(img_rows, img_cols, Nv_chan, acc_buffersiz, Na_chan)
model.summary()

opt = keras.optimizers.RMSprop()#(lr=3e-3)

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
    callbacks = [plot_learning,save_history]

# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)#,
        #shear_range=0.05,
        #zoom_range=0.5)

datagen.fit(xv_train)

class MySequence(keras.utils.Sequence):

    def __init__(self, xv_train, xa_train, y_train, datagen, batch_size):
        self.xv_train = xv_train
        self.xa_train = xa_train
        self.y_train = y_train
        self.datagen = datagen
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.xv_train) /
                       float(self.batch_size)).astype('int')

    def __getitem__(self, idx):
        i1 = idx * self.batch_size
        i2 = (idx+1) * self.batch_size
        batch_xv = self.xv_train[i1:i2, ]
        batch_xa = self.xa_train[i1:i2, ]
        batch_y = self.y_train[i1:i2, ]

        iterator = self.datagen.flow(batch_xv, batch_y,
                                     batch_size=self.batch_size,
                                     shuffle=False)
        batch_xv_aug, batch_y_aug = iterator.next()
        
        batch_xa_aug = np.copy(batch_xa)
        # Randoly change the magnitude
        mag = np.random.normal(loc=1.0, scale=0.1)
        batch_xa_aug *= mag

        # Randomly swap accelerometer channels
        acc1 = np.random.permutation([0, 1, 2])
        acc2 = np.random.permutation([3, 4, 5])
        batch_xa_aug[..., 0:3] = batch_xa_aug[..., acc1]
        batch_xa_aug[..., 3:6] = batch_xa_aug[..., acc2]
        
        return [batch_xv_aug, batch_xa_aug], batch_y

my_generator = MySequence(xv_train, xa_train, y_train,
                          datagen, batch_size=batch_size)

model.fit_generator(my_generator,
                    steps_per_epoch=xv_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=([xv_test, xa_test], y_test),
                    callbacks=callbacks, workers=16)

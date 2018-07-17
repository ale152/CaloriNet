from __future__ import division # Bluecrystal uses python 2.7
import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from PlotLearning import PlotLearning
from SaveHistory import SaveHistory
import numpy as np

from NetworkArchitectures import NetworkSilhouette
import CaloriesDataset

# Print this code on the console, useful for debugging
with open(__file__) as f: print('\n'.join(f.read().split('\n')[1:]))

Set = {'bluecrystal': False}

network_name = 'NETWORKNAME'

np.random.seed(10)

batch_size = 128
epochs = 1000

save_dir = os.path.join(os.getcwd(),'saved_models')

(x_train, y_train, lab_train) = CaloriesDataset.load_range(TRAINSUBJ)
(x_test, y_test, lab_test) = CaloriesDataset.load_range(TESTSUBJ)
Nchannels = x_train.shape[1]
img_rows = x_train.shape[2]
img_cols = x_train.shape[3]

# Reshape data into keras format
x_train = np.moveaxis(x_train,1,3)
x_test = np.moveaxis(x_test,1,3)

# Replace missing data with random sampling from same label
nan_train = np.isnan(y_train)
Nlabels = 11
for li in range(0,Nlabels+1):
    print('Replacing label %d' % li)
    
    tbr_train = np.all(np.column_stack((lab_train==li,nan_train)),1)
    Ntbr_tr = np.sum(tbr_train.astype('int'))
    # Randomly sample substitutes
    good_candidates_train = np.all(np.column_stack((lab_train==li,np.logical_not(nan_train))),1)
    Ngood_train = np.sum(good_candidates_train.astype('int'))
    print('%d elements to be replaced in train. Using %d good values' % (Ntbr_tr, Ngood_train))
    
    sub_i_tr = np.random.permutation(Ngood_train)
    # If there are more NaNs than good elements, simply sample them more than once
    if Ngood_train < Ntbr_tr:
        sub_i_tr = np.repeat(sub_i_tr,np.ceil(Ntbr_tr/Ngood_train))
        print('Not enough good elements for training. Data sampled %d times' % np.ceil(Ntbr_tr/Ngood_train))
    
    sub_i_tr = sub_i_tr[0:Ntbr_tr]
    good_candidates_train = np.take(np.where(good_candidates_train),sub_i_tr)
    y_train[tbr_train] = np.take(y_train,good_candidates_train)
    x_train[tbr_train,] = np.take(x_train,good_candidates_train,0)

# Remove Nan data from test
nan_test = np.isnan(y_test)
x_test = np.delete(x_test,np.where(nan_test),0)
y_test = np.delete(y_test,np.where(nan_test))

# Network
model = NetworkSilhouette(img_rows, img_cols, Nchannels)
model.summary()

opt = keras.optimizers.RMSprop(lr=0.003)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

x_train = x_train.astype('float32')/255;
x_test = x_test.astype('float32')/255;

best_model_name = '%s_augm_best.h5' % network_name

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
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
        #shear_range=0.05,
        #zoom_range=0.5)  # randomly flip images

datagen.fit(x_train)

# Debug data augmented images
#    for (img,cl) in datagen.flow(x_test,y_test,1):
#        plt.cla()
#        plt.imshow(img[0,:,:,0:3],extent=[0,1,0,1])
#        plt.axis('equal')
#        plt.pause(0.3)

model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
         steps_per_epoch=x_train.shape[0]//batch_size,
         epochs=epochs,
         validation_data=(x_test,y_test),
         callbacks=callbacks, workers=16)
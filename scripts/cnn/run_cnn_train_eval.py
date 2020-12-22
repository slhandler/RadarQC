import numpy as np
import pandas as pd
import xarray as xr
import h5py
import keras
import os
import calendar
import time
import datetime as dt
import matplotlib.pyplot as plt

from os.path import join
from itertools import product

from sklearn.isotonic import IsotonicRegression

# my routines
from utils import (findFiles, get_dataset_size, get_image_normalization_params)
from utils import (balanced_generator, generator, readHDF5,
                   combineFiles, normalize_images)
from utils import brier_skill_score_keras
from setup_cnn import build_cnn

from keras.metrics import AUC
from tensorflow.keras.models import load_model

import cnn_config as config

# define the output file and directory
outdir_and_file = config.OUTDIR_AND_FILE
target_variable = config.TARGET_VARIABLE

# path to normalization file (and filename as well)
norm_file = config.NORMALIZATION_FILE

# Define where the model will be saved
save_dir = config.MODEL_DIRECTORY
model_name = config.MODEL_NAME

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

# Define the input paths (just use RF data with target for getting the target field)
input_dir = config.PATH_TO_CNN_DATA
input_path_targets = config.PATH_TO_TARGETS

# define callback parameters and append to a list
list_of_callback_objects = []

checkpoint_object = keras.callbacks.ModelCheckpoint(
    filepath=model_path, monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='min',
    period=1)

list_of_callback_objects.append(checkpoint_object)

early_stopping_object = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0,
    patience=10, verbose=1, mode='min')

list_of_callback_objects.append(early_stopping_object)


# get list of training, calibration, and testing files
training_files = findFiles(
    input_dir, extension='.hdf5', start_date='20171001', end_date='20180331')
validation_files = findFiles(
    input_dir, extension='.hdf5', start_date='20161001', end_date='20161230')
testing_files = findFiles(input_dir, extension='.hdf5',
                          start_date='20170101', end_date='20170331')

# get list of training, calibration, and testing target files
training_target_files = findFiles(
    input_path_targets, extension='.csv', start_date='20171001', end_date='20180331')
validation_target_files = findFiles(
    input_path_targets, extension='.csv', start_date='20161001', end_date='20161230')
testing_target_files = findFiles(
    input_path_targets, extension='.csv', start_date='20170101', end_date='20170331')

# get size of training and validation sets
training_set_size = get_dataset_size(training_target_files)
validation_set_size = get_dataset_size(validation_target_files)
#testing_set_size    = get_dataset_size(testing_target_files)

# get the normalization params for training and validation
if (os.path.isfile(norm_file)):
    print("Reading coeffs for normalization...")
    normalization_dict = pd.read_csv(norm_file)
else:
    print("Computing coeffs for normalization...")
    normalization_dict = get_image_normalization_params(training_files)
    normalization_dict.to_csv(norm_file)

# define the CNN parameters
batch_size = 1024
n_epochs = 30
steps_per_epoch = np.floor(training_set_size/batch_size)
val_steps_per_epoch = np.ceil(validation_set_size/batch_size)
print(steps_per_epoch, val_steps_per_epoch)

# best params
nfeats = 5
dropout_rate = 0.25
l2_val = 0.0001
nfilters = 4*nfeats
filter_width = 3
learning_rate = 0.001

# construct the ConvNet
model = build_cnn(nfeats=nfeats,
                  nfilters=nfilters,
                  filter_width=filter_width,
                  dropout_rate=dropout_rate,
                  learning_rate=learning_rate,
                  l2_val=l2_val)

# define training and validation generators
training_generator = balanced_generator(training_files, training_target_files, batch_size,
                                        normalization_dict=normalization_dict)
validation_generator = generator(validation_files, validation_target_files, batch_size,
                                 normalization_dict=normalization_dict)

# fit model using the .fit_generator method
print("Training the model...")
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    verbose=2,
    workers=0,
    callbacks=list_of_callback_objects,
    validation_data=validation_generator,
    validation_steps=val_steps_per_epoch
)

# save the model
model.save(model_path)

# Read the files for validation into one larger array
print("Reading/Merging the validation files...")
validation_data = combineFiles(validation_files, extension='.hdf5')

print(validation_data.shape)

# scale the data appropriately
print("Scaling the data...")
predictor_matrix, _ = normalize_images(
    predictor_matrix=validation_data[:, :, :, :],
    normalization_dict=normalization_dict)

predictor_matrix = predictor_matrix.astype('float32')
print("Data has been scaled....")

print("Reading/Merging validation targets...")
targets = combineFiles(validation_target_files, extension='.csv')
print(targets.shape)

# get the data for targets from the dataframe
calibration_target_labels = targets[target_variable].to_numpy().astype(
    np.float32)

# now, we can make predictions
print("Making predictions for Isotonic Regression...")
probs = model.predict(predictor_matrix, batch_size=1000)[:, 0]
probs = probs.astype(np.float32)

# now take probs and train an isotonic regression model
print("Training Isotonic Regression Model...")
model_isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
model_isotonic.fit(probs, calibration_target_labels)
print("DONE TRAINING THE ISOTONIC CALIBRATION MODEL")

# delete variables to save space
del validation_data
del predictor_matrix

# lastly, evaluate the model and write out to file!
# Read the files for validation into one larger array
print("Reading/Merging the testing files...")
testing_data = combineFiles(testing_files, extension='.hdf5')

print(testing_data.shape)

# scale the data appropriately
print("Scaling the data...")
predictor_matrix, _ = normalize_images(
    predictor_matrix=testing_data[:, :, :, :],
    normalization_dict=normalization_dict)

predictor_matrix = predictor_matrix.astype('float32')
print("Data has been scaled...")

print("Reading/Merging testing targets...")
targets = combineFiles(testing_target_files, extension='.csv')
print(targets.shape)

# now, we can make predictions
print("Making testing predictions...")
probs = model.predict(predictor_matrix, batch_size=1000)[:, 0]
probs = probs.astype(np.float32)

print("Calibrating the testing set probs...")
calibrated_probs = model_isotonic.predict(probs)

# fin nans/inf issues
infin = np.where(~np.isfinite(probs))
if (len(infin[0]) > 0):
    probs[infin] = 0.0

infin = np.where(~np.isfinite(calibrated_probs))
if (len(infin[0]) > 0):
    calibrated_probs[infin] = 0.0

test_df_out = targets.copy()
test_df_out.loc[:, f'prob_{target_variable}'] = probs
test_df_out.loc[:, f'calibrated_prob_{target_variable}'] = calibrated_probs

# Finally, write out the file
print("Writing out final file.....")
#test_df_out.dropna(inplace=True, how='any', axis=0)
test_df_out.reset_index(drop=True, inplace=True)
test_df_out.to_csv(outdir_and_file)
print("DONE!!!")

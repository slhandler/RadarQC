import numpy as np
import pandas as pd
import xarray as xr
import h5py
import keras
import os
import calendar
import time
import datetime as dt
from os.path import join
from itertools import product
import pickle

from sklearn.metrics import roc_auc_score, average_precision_score

from utils import (findFiles, get_dataset_size, get_image_normalization_params)
from utils import (balanced_generator, generator, readHDF5)
from setup_cnn import build_cnn_tuner

from kerastuner.tuners import BayesianOptimization
import kerastuner
from keras.metrics import AUC

import cnn_config as config

# path to normalization file (and filename as well)
norm_file = config.NORMALIZATION_FILE
log_dir = config.LOG_DIR

# Define where the model will be saved
save_dir = config.MODEL_DIRECTORY
model_name = config.MODEL_NAME

if not os.path.isdir(save_dir): os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

#Define the input paths (just use RF data with target for getting the target field)
input_dir = config.PATH_TO_CNN_DATA
input_path_targets = config.PATH_TO_TARGETS

#get list of training and validation files
training_files   = findFiles(input_dir, extension='.hdf5', start_date='20171001', end_date='20180331')
validation_files = findFiles(input_dir, extension='.hdf5', start_date='20161201', end_date='20161230')

#get list of training and validation target files
training_target_files   = findFiles(input_path_targets, extension='.csv', start_date='20171001', end_date='20180331')
validation_target_files = findFiles(input_path_targets, extension='.csv', start_date='20161201', end_date='20161230')

#get size of training and validation sets
training_set_size   = get_dataset_size(training_target_files)
validation_set_size = get_dataset_size(validation_target_files)

# define parameters for generator
batch_size          = 512
n_epochs            = 32
steps_per_epoch     = training_set_size // batch_size
val_steps_per_epoch = validation_set_size // batch_size

#define callback parameters and append to a list
list_of_callback_objects = []

checkpoint_object = keras.callbacks.ModelCheckpoint(
           filepath=model_path, monitor='val_auc_roc', verbose=1,
           save_best_only=True, save_weights_only=False, mode='max',
           period=1)

list_of_callback_objects.append(checkpoint_object)

early_stopping_object = keras.callbacks.EarlyStopping(
       monitor='val_auc_roc', min_delta=0.0,
       patience=5, verbose=1, mode='max')

list_of_callback_objects.append(early_stopping_object)

#get the normalization params for training and validation
if (os.path.isfile(norm_file)):
    print("Reading coeffs for normalization...")
    normalization_dict = pd.read_csv(norm_file)
else:
    print("Computing coeffs for normalization...")
    normalization_dict = get_image_normalization_params(training_files)
    normalization_dict.to_csv(norm_file)

#define training and validation generators
training_generator = balanced_generator(training_files, training_target_files, batch_size, 
                            normalization_dict=normalization_dict)
validation_generator = generator(validation_files, validation_target_files, batch_size, 
                            normalization_dict=normalization_dict)

# create instance of the Bayesian Optimization routine
tuner = BayesianOptimization(
    build_cnn_tuner,
    objective = kerastuner.Objective("val_auc_roc", direction="max"),
    metrics = [AUC(curve="PR", name='auc_pr'), AUC(curve="ROC", name='auc_roc')],
    max_trials = 10,
    seed = 42,
    executions_per_trial = 1,
    directory = LOG_DIR
)

# print summary of the search space
print(tuner.search_space_summary())

# run through the search
tuner.search(
    training_generator,
    steps_per_epoch=steps_per_epoch,
    epochs = n_epochs,
    validation_data = validation_generator,
    validation_steps = val_steps_per_epoch,
    verbose=2,
    callbacks=list_of_callback_objects
)

print("Done searching...")
# print the summary
print(tuner.results_summary())

best_three_models = tuner.get_best_models(num_models=3)

print("Best three models:")
print(best_three_models)

best_hp = tuner.get_best_hyperparameters()[0]

print("Best hyperparams:")
print(best_hp)

import pandas as pd
import numpy as np
import os
import sys
import joblib

# custom imports for handlings sampling and splitting data
from RadarQC.ml_lib.core.sample_methods import SampleTechniques
from RadarQC.ml_lib.core.split_data import DataSplitter
from RadarQC.ml_lib.utils.utils_funcs import splitTargetFromLabels

# config file storing all necessary inputs for script
import radar_qc_config as config

# config file storing hyperparameter information for any ML model
import ml_hyperparameter_config as hyperparam_config

# scikit learn imports
from skopt import BayesSearchCV
from skopt import gp_minimize, space

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from functools import partial
from sklearn.model_selection import cross_val_score

# get config params
predictor_columns = config.PREDICTOR_COLUMNS
target_column = config.TARGET_COLUMN
train_data_file = config.TRAINING_FILENAME
test_data_file = config.TESTING_FILENAME
scoring_func = config.SCORING_FUNC
hyperparm_n_iter = config.HYPER_PARAM_N_ITER
output_filename = config.LR_FILENAME

print("Reading in training DataFrame")
train_data = pd.read_csv(train_data_file)
print(train_data.shape)

# split predictors from label
train_predictors, train_targets = splitTargetFromLabels(train_data,
                                                        target_column=target_column)

# make an instance of dataSplitter class
splitData = DataSplitter(train_predictors, train_targets, target_column)

# get monthly splits
cv_dict = splitData.splitByMonth()
cv_splits = list(zip(cv_dict['train_indices'], cv_dict['test_indices']))

# tune RF using Bayes over monthly cv splits
# set up a variable transformation (with standardization)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
lr = LogisticRegression(penalty='elasticnet', solver='saga',
                        class_weight='balanced', random_state=42)

# set up a pipeline for hyperparameter tuning
lr_pipeline = Pipeline(steps=[('scaler', pt), ('lr', lr)])

param_search_space = hyperparam_config.hyper_dict.get(
    'lr_hyperparameter_space')

opt = BayesSearchCV(lr_pipeline,
                    param_search_space,
                    cv=cv_splits, n_iter=hyperparm_n_iter, scoring=scoring_func,
                    refit=True)

print("Perfoming hyperparameter search...")
opt.fit(train_predictors[predictor_columns], train_targets)

print("Best performance...")
print(opt.best_score_)
print(opt.best_params_)

# build best model
lr_model = opt.best_estimator_
print(lr_model)

# pull out the data transformation piece from the model
pt = lr_model[0]
lr_model = lr_model[1]

print("Saving model...")
joblib.dump(lr_model, config.LR_MODEL_PATH)

print("Reading in calibration/evaluation DataFrame")
data = pd.read_csv(test_data_file)
data.dropna(inplace=True, how='any', axis=0)
data.reset_index(drop=True, inplace=True)
print(data.shape)

# make an instance of dataSplitter class
splitData = DataSplitter(data, data[target_column].to_numpy(), target_column)

# split dataset into calibrate and testing sets
calibrate_data = splitData.splitByDate(
    date_to_split='2016-12-30 23:50', is_greater=False)
testing_data = splitData.splitByDate(
    date_to_split='2017-01-01 00:00', is_greater=True)

# split predictors from labels for calibration set
calibration_predictors, calibration_targets = splitTargetFromLabels(calibrate_data,
                                                                    target_column=target_column)

# split predictors from labels for testing set
testing_predictors, testing_targets = splitTargetFromLabels(testing_data,
                                                            target_column=target_column)

# train isotonic regression model
scaled_cali_predictors = pt.transform(
    calibration_predictors.loc[:, predictor_columns])
scaled_cali_predictors = pd.DataFrame(
    scaled_cali_predictors, columns=predictor_columns)
input_predictions = lr_model.predict_proba(scaled_cali_predictors)[:, 1]

print("Training an isotonic regression model...")
isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
isotonic_model.fit(input_predictions, calibration_targets)

print("Saving calibration model...")
joblib.dump(isotonic_model, config.LR_ISO_MODEL_PATH)

# evaluate final model using the test set
print("Evaluating test set...")
scaled_test_predictors = pt.transform(
    testing_predictors.loc[:, predictor_columns])
scaled_test_predictors = pd.DataFrame(
    scaled_test_predictors, columns=predictor_columns)

raw_probs = lr_model.predict_proba(scaled_test_predictors)[:, 1]
calibrated_probs = isotonic_model.predict(raw_probs)

# fix any nan predictions
infin = np.where(~np.isfinite(calibrated_probs))
if (len(infin[0]) > 0):
    calibrated_probs[infin] = 0.0

# assign probabilities to a csv file to pefrom future analysis with
testing_data.loc[:, f'calibrated_prob_{target_column}'] = calibrated_probs
testing_data.loc[:, f'prob_{target_column}'] = raw_probs

# write out for graphical analysis
testing_data.to_csv(output_filename)
print("DONE!")

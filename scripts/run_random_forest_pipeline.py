import os
from functools import partial
import joblib
import numpy as np
import pandas as pd

# custom imports for handlings sampling and splitting data
from RadarQC.ml_lib.core.sample_methods import SampleTechniques
from RadarQC.ml_lib.core.split_data import DataSplitter
from RadarQC.ml_lib.utils.utils_funcs import splitTargetFromLabels

# scikit learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV, gp_minimize, space

# config file storing hyperparameter information for any ML model
import ml_hyperparameter_config as hyperparam_config
# config file storing all necessary inputs for script
import radar_qc_config as config


def optimize_skopt(params, param_names, examples, targets, splits=3, scoring_func='roc_auc'):
    """Funciton called to optimize hyperparameters

    Args:
        params (list): list of values for each hyperparameter
        param_names (list): hyperparameter names to tune
        examples (pd.DataFrame): predictors for model
        targets (np.ndarray): array of targets
        splits (int, dict, optional): If int, represents the number of splits.
            If dict, mapping for train-test split indices.
        scoring_func (str, optional): scoring function to use. Defaults to 'roc_auc'.

    Returns:
        (float): loss for given set of hyperparameters
    """

    # make dict using two lists
    params = dict(zip(param_names, params))

    # define model
    model = RandomForestClassifier(random_state=42, criterion='entropy',
                                   class_weight='balanced', **params)

    # perform cross validation over my splits
    cv_scores = cross_val_score(
        model, examples, targets, cv=splits, scoring=scoring_func)

    # return the mean score
    return -1.0*np.mean(cv_scores)


# get config params
predictor_columns = config.PREDICTOR_COLUMNS
target_column = config.TARGET_COLUMN
train_data_file = config.TRAINING_FILENAME
test_data_file = config.TESTING_FILENAME
scoring_func = config.SCORING_FUNC
hyperparm_n_iter = config.HYPER_PARAM_N_ITER
output_filename = config.RF_FILENAME

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

# setup hyperparameter search space
rf_hyperparameter_space = [
    space.Integer(5, 100, name='min_samples_split'),
    space.Integer(5, 100, name='min_samples_leaf'),
    space.Integer(100, 500, name='n_estimators'),
    space.Integer(3, 25, name='max_depth'),
    space.Real(0.01, 1, prior='uniform', name='max_features')
]

param_names = ['min_samples_split', 'min_samples_leaf', 'n_estimators',
               'max_depth', 'max_features']

optimization_function = partial(
    optimize_skopt,
    param_names=param_names,
    examples=train_predictors[predictor_columns],
    targets=train_targets,
    splits=cv_splits,
    scoring_func=scoring_func
)

# tune RF using Bayes over monthly cv splits
result = gp_minimize(
    optimization_function,
    dimensions=rf_hyperparameter_space,
    n_calls=hyperparm_n_iter,
    n_initial_points=10,
    verbose=10,
    n_jobs=3
)

# extract the best set of hyperparameters as a dictionary
best_param_dict = dict(zip(param_names, result.x))

# build best model
rf_model = RandomForestClassifier(random_state=42, criterion='entropy',
                                  class_weight='balanced', **best_param_dict)
print(rf_model)

print("Saving model...")
joblib.dump(rf_model, config.RF_MODEL_PATH)

# train on whole training set
print("Training the final model...")
rf_model.fit(train_predictors[predictor_columns], train_targets)

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
print("Training an isotonic regression model...")
input_predictions = rf_model.predict_proba(
    calibration_predictors[predictor_columns])[:, 1]
isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
isotonic_model.fit(input_predictions, calibration_targets)

print("Saving calibration model...")
joblib.dump(isotonic_model, config.RF_ISO_MODEL_PATH)

# evaluate final model using the test set
print("Evaluating test set...")
raw_probs = rf_model.predict_proba(testing_predictors[predictor_columns])[:, 1]
calibrated_probs = isotonic_model.predict(raw_probs)

# fix any nan predictions
infin = np.where(~np.isfinite(calibrated_probs))
if (len(infin[0]) > 0):
    calibrated_probs[infin] = 0.0

# assign probabilities to a csv file to pefrom future analysis with
testing_data.loc[:, f'calibrated_prob_{target_column}'] = calibrated_probs
testing_data.loc[:, f'prob_{target_column}'] = raw_probs

# write out predictions for analysis
testing_data.to_csv(output_filename)
print("DONE!")

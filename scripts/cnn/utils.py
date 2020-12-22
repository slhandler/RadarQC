import numpy as np
import pandas as pd
import h5py
import os
import datetime as dt

import keras.backend as K
from keras.utils import np_utils


def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)


def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    return 1.0 - brier_score_keras(obs, preds) / climo


def _update_normalization_params(intermediate_dict, new_values, pred_index):
    """Updates normalization params for a predictor."""

    if (intermediate_dict is None):
        zero_data = np.zeros(shape=(5, 2))
        intermediate_dict = pd.DataFrame(zero_data, columns=['Mean', 'Std'])

    intermediate_dict.at[pred_index, 'Mean'] = np.nanmean(new_values)
    intermediate_dict.at[pred_index, 'Std'] = np.nanstd(new_values, ddof=1)

    return intermediate_dict


def get_dataset_size(list_of_files):
    """Given a list of files, this will return the cumulative size of the dataFrame."""

    dataset_size = 0

    for file in list_of_files:

        data = pd.read_csv(file)

        dataset_size += data.shape[0]

    return dataset_size


def get_image_normalization_params(netcdf_file_names):
    """Computes normalization params (mean and stdev) for each predictor.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    normalization_dict = None

    for this_file_name in netcdf_file_names:

        #print("Reading data from: " + this_file_name)
        this_data = readHDF5(this_file_name)

        for m in range(this_data.shape[3]):
            normalization_dict = _update_normalization_params(
                intermediate_dict=normalization_dict,
                new_values=this_data[:, :, :, m],
                pred_index=m)

    return normalization_dict


def normalize_images(predictor_matrix, normalization_dict=None):
    """Normalizes images to z-scores."""

    num_predictors = 5

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = np.nanmean(predictor_matrix[..., m])
            this_stdev = np.nanstd(predictor_matrix[..., m], ddof=1)

            normalization_dict[predictor_names[m]] = np.array(
                [this_mean, this_stdev])

    for m in range(num_predictors):
        this_mean = normalization_dict.at[m, 'Mean']
        this_stdev = normalization_dict.at[m, 'Std']

        predictor_matrix[..., m] = (
            (predictor_matrix[..., m] - this_mean) / float(this_stdev)
        )

    return predictor_matrix, normalization_dict


def readHDF5(input_directory_and_file):
    """ Given a input director and filename, open and read the hdf5 file.
        Save/return as a numpy n-D array."""

    h5_file = h5py.File(input_directory_and_file, 'r')
    array_of_data = np.array(h5_file.get('dataset'))

    h5_file.close()

    return array_of_data


def findFiles(input_file_directory, extension='.hdf5', start_date='20161001', end_date='20170331'):
    """ This routine returns a list of file names to read in future steps, given
    a date range. """

    DATE_FORMAT = '%Y%m%d'

    list_of_files = []
    all_file_times = []

    start_date = dt.datetime.strptime(start_date, DATE_FORMAT)
    end_date = dt.datetime.strptime(end_date, DATE_FORMAT)

    all_files = sorted([f for f in os.listdir(
        input_file_directory) if f.endswith(extension)])

    for file in all_files:
        all_file_times.append(file.split('.')[0])

    dates_array = np.array([dt.datetime.strptime(date, '%Y%m%d')
                            for date in all_file_times])

    igood = np.where((dates_array >= start_date) & (dates_array <= end_date))

    list_of_files = np.array(all_files)[igood[0]]

    final_file_list = []

    for ifile in list_of_files:
        final_file_list.append(input_file_directory + ifile)

    return final_file_list


def combineFiles(list_of_files, extension='.hdf5'):
    """ This routine will take a list of input files and merge them into an array """

    if (extension == '.hdf5'):
        data_matrix = None
    else:
        data_matrix = pd.DataFrame()

    for file in list_of_files:

        #print('\n Reading data from: "{0:s}"...'.format(file))

        if (extension == '.hdf5'):
            this_data = readHDF5(file)

            if ((data_matrix is None) or (data_matrix.size == 0)):
                data_matrix = this_data
            else:
                data_matrix = np.concatenate(
                    [data_matrix, this_data], axis=0)

        else:
            this_data = pd.read_csv(file)

            if ((data_matrix is None) or (data_matrix.shape[0] == 0)):
                data_matrix = this_data
            else:
                data_matrix = pd.concat([data_matrix, this_data])

    if (extension == '.csv'):
        data_matrix.reset_index(drop=True, inplace=True)

    return data_matrix


def generator(input_file_list, target_file_list, num_examples_per_batch,
              normalization_dict=None):

    # randomly shuffle the input files
    lf = list(zip(input_file_list, target_file_list))
    np.random.shuffle(lf)

    input_file_list, target_file_list = zip(*lf)

    num_files = len(input_file_list)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None

    while True:

        while num_examples_in_memory < 2*num_examples_per_batch:

            #print('\n Reading data from: "{0:s}"...'.format(input_file_list[file_index]))

            this_array = readHDF5(input_file_list[file_index])
            this_target = pd.read_csv(target_file_list[file_index])

            # check to make sure there is some precip data, else skip
            if (this_target['valid_precip'].mean() <= 0.000000):
                file_index += 1
                if (file_index >= num_files):
                    file_index = 0
                continue

            the_indices = this_target.index.to_list()

            file_index += 1
            if (file_index >= num_files):
                file_index = 0

            if ((full_target_matrix is None) or (full_target_matrix.size == 0)):
                full_predictor_matrix = this_array[the_indices, :, :, :]
                full_target_matrix = this_target.loc[the_indices,
                                                     'valid_precip'].values
            else:
                full_predictor_matrix = np.concatenate(
                    [full_predictor_matrix, this_array[the_indices, :, :, :]], axis=0)
                full_target_matrix = np.concatenate(
                    (full_target_matrix, this_target.loc[the_indices, 'valid_precip'].values), axis=0)

            num_examples_in_memory = full_predictor_matrix.shape[0]

        batch_indices = np.linspace(0, num_examples_in_memory - 1,
                                    num=num_examples_in_memory, dtype=int)
        batch_indices = np.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        # scale the data appropriately
        predictor_matrix, _ = normalize_images(
            full_predictor_matrix[batch_indices, :, :, :],
            normalization_dict=normalization_dict)

        predictor_matrix = predictor_matrix.astype('float32')

        # get correct target values
        target_matrix = full_target_matrix[batch_indices]

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_matrix)


def balanced_generator(input_file_list, target_file_list, num_examples_per_batch,
                       normalization_dict=None):

    # randomly shuffle the input files
    lf = list(zip(input_file_list, target_file_list))
    np.random.shuffle(lf)

    input_file_list, target_file_list = zip(*lf)

    num_files = len(input_file_list)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None

    # counters to make sure classes will have enough examples to sample afte while loop
    num_ones = 0
    num_zeros = 0

    while True:

        while ((num_examples_in_memory < num_examples_per_batch) or
               (num_ones < int(num_examples_per_batch/2.)) or
               (num_zeros < int(num_examples_per_batch/2.))):

            this_array = readHDF5(input_file_list[file_index])
            this_target = pd.read_csv(target_file_list[file_index])

            # check to make sure there is some precip data, else skip
            if (this_target['valid_precip'].mean() <= 0.000000):
                file_index += 1
                if (file_index >= num_files):
                    file_index = 0
                continue

            # get the number of ones and zeros
            n_ones = this_target['valid_precip'].value_counts()[1]
            n_zeros = this_target['valid_precip'].value_counts()[0]

            if (n_ones <= n_zeros):
                # get equal matching ones and zeros
                the_zeros = this_target[this_target['valid_precip'] == 0].sample(n=n_ones,
                                                                                 replace=False).index.to_list()
                the_ones = np.where(this_target['valid_precip'] > 0)[0]
            else:
                # get equal matching ones and zeros
                the_zeros = this_target[this_target['valid_precip'] == 0].sample(
                    n=n_zeros, replace=False).index.to_list()
                the_ones = np.where(this_target['valid_precip'] > 0)[0]

            # increment counters based on data from file
            num_ones += the_ones.shape[0]
            num_zeros += len(the_zeros)

            # concatenate indices and sort in ascending order
            the_indices = np.sort(np.concatenate(
                (the_zeros, the_ones), axis=None))

            file_index += 1
            if (file_index >= num_files):
                file_index = 0

            if ((full_target_matrix is None) or (full_target_matrix.size == 0)):
                full_predictor_matrix = this_array[the_indices, :, :, :]
                full_target_matrix = this_target.loc[the_indices,
                                                     'valid_precip'].values
            else:
                full_predictor_matrix = np.concatenate(
                    [full_predictor_matrix, this_array[the_indices, :, :, :]], axis=0)
                full_target_matrix = np.concatenate(
                    (full_target_matrix, this_target.loc[the_indices, 'valid_precip'].values), axis=0)

            num_examples_in_memory = full_predictor_matrix.shape[0]

        # get equal 50-50 splits
        batch_indices_ones = np.random.choice(np.where(full_target_matrix > 0)[0],
                                              size=int(num_examples_per_batch/2.), replace=False)

        batch_indices_zeros = np.random.choice(np.where(full_target_matrix < 1)[0],
                                               size=int(num_examples_per_batch/2.), replace=False)

        batch_indices = np.sort(np.concatenate(
            (batch_indices_zeros, batch_indices_ones), axis=None))

        # scale the data appropriately
        predictor_matrix, _ = normalize_images(
            full_predictor_matrix[batch_indices, :, :, :],
            normalization_dict=normalization_dict)

        predictor_matrix = predictor_matrix.astype('float32')

        # get correct target values
        target_matrix = full_target_matrix[batch_indices]

   #     print('Fraction of examples in positive class: ', np.mean(target_matrix))

        num_examples_in_memory = 0
        num_ones = 0
        num_zeros = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_matrix)

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

import keras.backend as K
from keras.utils import np_utils
from tensorflow.keras.metrics import AUC

from kerastuner.engine.hyperparameters import HyperParameters

from utils import brier_skill_score_keras


def build_cnn_tuner(hp):
    ''' This function can be used for tuning a convnet using the kerastuner package.
        Inputs
            hp: A HyperParameters instance.
     '''

    # define the sequential model
    conv_model = Sequential()

    # First set of 2D convolution layers; followed by regularization
    conv_model.add(Conv2D(filters=8,
                          kernel_size=(3, 3),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True, input_shape=(33, 33, 5),
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))

    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=hp.Float(
        'alpha', 0.1, 0.5, sampling='linear')))
    conv_model.add(MaxPooling2D())

    # Second set of 2D convolution layers; followed by regularization
    conv_model.add(Conv2D(filters=16,
                          kernel_size=(3, 3),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))

    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=hp.Float(
        'alpha', 0.1, 0.5, sampling='linear')))
    conv_model.add(MaxPooling2D())

    # Third set of 2D convolution layers; followed by regularization
    conv_model.add(Conv2D(filters=32,
                          kernel_size=(3, 3),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))

    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=hp.Float(
        'alpha', 0.1, 0.5, sampling='linear')))
    conv_model.add(MaxPooling2D())

    # Flatten the last convolutional layer into a long feature vector
    conv_model.add(Flatten())

    # 1st dense output layer which will reduce the output size to 49
    conv_model.add(Dense(524, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=hp.Float(
        'alpha', 0.1, 0.5, sampling='linear')))
    conv_model.add(Dropout(rate=hp.Float(
        'dropout_rate', 0.2, 0.5, sampling='linear')))

    # 2nd dense output layer which will reduce the output size to 49
    conv_model.add(Dense(49, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=hp.Float(
        'alpha', 0.1, 0.5, sampling='linear')))
    conv_model.add(Dropout(rate=hp.Float(
        'dropout_rate', 0.2, 0.5, sampling='linear')))

    # Final dense output layer, equivalent to a logistic regression on the last layer
    conv_model.add(Dense(1, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2(hp.Float('l2_val', 0.001, 0.1, sampling='log'))))
    conv_model.add(Activation("sigmoid"))

    # Use the Adam optimizer
    optimizer = Adam(lr=hp.Float('learning_rate', 0.001, 0.1, sampling='log'))
    conv_model.compile(optimizer=optimizer, loss="binary_crossentropy",
                       metrics=[AUC(curve="PR", name='auc_pr'), AUC(curve="ROC", name='auc_roc')])
    print(conv_model.summary())

    return conv_model


def build_cnn(nfeats=1, nfilters=10, filter_width=3, learning_rate=0.001,
              alpha=0.3, dropout_rate=None, l2_val=None):
    ''' This function builds the architecture for the final convnet.
        Inputs:
            nfeats : (int) number of input features
            nfilters : (int) number of filters per convolution layer
            filter_width : (int) width of filter
            learning_rate : (float) learning rate for optimization algorithm
            alpha : (float) for activation function
            dropout_rate : (float) percentage of dense units set to zero
            l2_val : (float) value for L2 regularization
     '''

    if l2_val is None:
        l2_reg = None
    else:
        l2_reg = l2(l2_val)

    # define the sequential model
    conv_model = Sequential()

    # First set of 2D convolution layers; followed by regularization
    conv_model.add(Conv2D(filters=nfilters,
                          kernel_size=(filter_width, filter_width),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True, input_shape=(33, 33, nfeats),
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))

    conv_model.add(Conv2D(filters=nfilters*2,
                          kernel_size=(filter_width, filter_width),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))
    conv_model.add(MaxPooling2D())

    # Second set of 2D convolution layers; followed by regularization
    conv_model.add(Conv2D(filters=nfilters*4,
                          kernel_size=(filter_width, filter_width),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))

    conv_model.add(Conv2D(filters=nfilters*8,
                          kernel_size=(filter_width, filter_width),
                          strides=(1, 1), padding="same", data_format="channels_last",
                          activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))
    conv_model.add(MaxPooling2D())

    # Flatten the last convolutional layer into a long feature vector
    conv_model.add(Flatten())

    # 1st dense output layer which will reduce the output size to 49
    conv_model.add(Dense(512, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))
    conv_model.add(Dropout(rate=dropout_rate))

    # 2nd dense output layer which will reduce the output size to 49
    conv_model.add(Dense(49, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2_reg))
    conv_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    conv_model.add(LeakyReLU(alpha=alpha))
    conv_model.add(Dropout(rate=dropout_rate))

    # Dense output layer, equivalent to a logistic regression on the last layer
    conv_model.add(Dense(1, use_bias=True, activation=None,
                         kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=l2_reg))
    conv_model.add(Activation("sigmoid"))

    # Use the Adam optimizer
    optimizer = Adam(lr=learning_rate)
    conv_model.compile(optimizer=optimizer, loss="binary_crossentropy",
                       metrics=[AUC(curve="PR", name='auc_pr'), AUC(curve="ROC", name='auc_roc'),
                                brier_skill_score_keras])

    print(conv_model.summary())

    return conv_model

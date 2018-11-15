#!/usr/bin/python3
# coding: utf-8


## Official librairies
import sys
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.utils import to_categorical


import sklearn
from sklearn.model_selection import train_test_split


## Custom librairies
from lib.Mathlib import Create_Df, FromCtoN, NN_model, plotvars, selectKbest, LearningCurve, Compare_activation
from lib.Mathlib import Sgd, Rms, Adagrad, Adadelta, Adam, Adamax, Nadam, parameter_tuning

if __name__ == '__main__':
    ## Import data
    df_X = Create_Df(sys.argv[1])
    X = df_X.iloc[1:, 1:]
    df_Y = Create_Df(sys.argv[2])
    Y_cat = df_Y.iloc[1:, 1:]

    # Return values for categorical labels
    Y = FromCtoN(Y_cat)
    # A brief statistical study
    plotvars(X, Y, df_X)

    # Select K best features according to chi2 stat of non negative features
    X_red, X_filtered = selectKbest(X, Y, df_X, k=15)

    ## Initiialize model cross_validation
    # Convert data to categorical for binary-crossentropy calculation
    Y_bis = to_categorical(Y)
    # Split the model in training and test sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_red, Y_bis, test_size = 0.33, random_state = 42)

    ## Parameters comparison part
    study = 0
    yes = ['y', 'Y', 'Yes', 'YES']
    no = ['n', 'no', 'N', 'NO']
    while study not in yes and study not in no:
        study = input('Do you want to execute parameters comparison? (y/n) ')
    if study in yes:
        param = -1
        while param != '0' and param != '1' and param != '2':
            param = input('\n1: Launch activation functions comparison\n2: Launch optimizers comparison\n0: Launch both\nYour choice: ')
        if param == '0' or param == '1':
            ## Activation comparison
            all_activations = ['linear', 'sigmoid', 'tanh', 'softmax', 'softsign', 'softplus']
            default_optimizers = {'SGD':optimizers.SGD(), 'RMSprop':optimizers.RMSprop(), 'Adagrad':optimizers.Adagrad(), 'Adadelta':optimizers.Adadelta(), 'Adam':optimizers.Adam(), 'Adamax':optimizers.Adamax(), 'Nadam':optimizers.Nadam()}
            for opti in default_optimizers:
                activation_histories = [NN_model(X_train, X_test, y_train, y_test, optimizer=default_optimizers[opti] , activation=i) for i in all_activations]
                Compare_activation(activation_histories, all_activations, opti)

        if param == '0' or param == '2':
            ## Optimizers comparison
            all_optimizers = ['sgd', 'rms', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
            activations_selected = ['softplus']
            parameter_tuning(X_train, X_test, y_train, y_test, all_optimizers, activations_selected)

    ## The model and learning curves
    # Create and train a neural network model
    adagrad = Adagrad(0.01)
    history = NN_model(X_train, X_test, y_train, y_test, optimizer=adagrad , activation='softplus')

    ## Show learning curves for the model
    LearningCurve(history, 'Activation : Softplus, Optimizer = Adagrad, Learning rate = 0.01')

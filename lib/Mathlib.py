#!/usr/bin/python3
# coding: utf-8

### Official librairies
import pandas as pd
import numpy as np
from numpy import argmax
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import pdist

import keras
import keras.backend as K
from keras import optimizers, activations
from keras.utils import to_categorical, np_utils
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model

import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, ShuffleSplit, cross_val_score, learning_curve, validation_curve
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.cluster import AgglomerativeClustering, k_means
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, auc, roc_curve, roc_auc_score, mean_squared_error, auc
from sklearn.metrics import precision_recall_fscore_support as Score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier


def Create_Df(path):
    '''Create the pandas dataframe.
       path: string: path of the file to read into the dataframe.
    return: df: pandas dataframe
    '''
    df = pd.read_csv(path, header=None)

    return df


def FromCtoN(Y):
    '''Return values for categorical labels.
       Y: pandas dataframe: labels
    return: Y_encoded: list: labels values
    '''
    Y_encoded = []
    # Pull labels
    for i in Y.iloc[:,0] :
        if i == 'BRCA' :
            Y_encoded.append(0)
        if i == 'KIRC' :
            Y_encoded.append(1)
        if i == 'PRAD' :
            Y_encoded.append(2)
        if i == 'LUAD' :
            Y_encoded.append(3)
        if i == 'COAD' :
            Y_encoded.append(4)
    # Compute label weights
    occurences = [Y_encoded.count(i)/len(Y_encoded) for i in range(5)]

    return Y_encoded


def getVar(df, param):
    '''Get variances.
       df: pandas dataframe: data
       param: parameter for variance calculation
    return: vars: list: variances
    '''
    start = param[0]
    end = param[1]

    if param == 'max':
        start = 0
        end = df.shape[1]

    vars = [np.var([j for j in df.iloc[:,col] if type(j)==float]) for col in range(start, end)]

    return vars


def NN_model(X_train, X_test, y_train, y_test, optimizer, activation):
    '''Create and train a neural network model with data.
       X_train: data to train the model (expression levels)
       y_train: data to train the model (labels)
       X_test: data to try and predict the labels with the model
       y_test: data to compare with the predictions and see the accuracy of our model
       optimizer: optimize the regularization
    return: history: keras object representing the model
    '''
    # Initialize model
    init = 'random_uniform'

    # Define topology
    L1 = Input(shape = (15,))
    L3 = Dense(7,activation = activation, kernel_initializer = init)(L1)
    L4 = Dense(5, activation = 'softmax',kernel_initializer = init)(L3)

    # Create model
    model = Model(input = L1, output = L4)

    # Compile model
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fit the model with datas
    history = model.fit(X_train, y_train, batch_size=64, epochs = 100, verbose = 0, validation_split=0.33, shuffle=True)

    # Prediction
    y_pred = model.predict(X_test)

    # from categorical to vector:
    y_test = y_test.argmax(axis=-1)  # Label
    y_pred = y_pred.argmax(axis=-1)


    # Draw the confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['labels'], colnames=['predictions'])
    print(confusion_matrix)

    return history


def plotvars(X, Y, df):
    '''Plot two graphics for the variance.
       X: pandas dataframe: data
       Y: pandas dataframe: labels
       df: pandas dataframe: raw data
    Show: a graph with all features variances
          a graph with variances means for k different numbers of features selected

    '''
    plt.figure(figsize=(16,16))
    # Plot vars
    plt.subplot(211)
    # Set parameter for variance calculation (which column to use)
    param = 'max'
    # Pull variance from the getVar fonction
    vars = getVar(X, param)

    plt.hist(vars, bins = 'auto')
    plt.ylabel("Counts")
    plt.title('Feature variances')

    # select features for which variances are upper than 1
    selected_features = [i for i in vars if i > 1]

    plt.subplot(212)
    means_X_filtered_vars = []
    nb_features = [5, 9, 20, 50, 100, 1000, 4000, len(selected_features)]
    for i in nb_features:
        X_red, X_filtered = selectKbest(X, Y, df, k=i)
        Vars = getVar(X_filtered, param)
        means_X_filtered_vars.append(np.mean(Vars))
    plt.plot(nb_features, [j for j in means_X_filtered_vars])
    plt.title('Variances means for k different numbers of features selected')
    plt.ylabel("Variance mean")
    plt.savefig('output/variances.png')
    plt.show()


def selectKbest(X, Y, df_X, k):
    '''Select the k best features in our data using chi2 test: the k less independant ones.
       X: pandas dataframe: data
       Y: pandas dataframe: labels
       df_X: pandas dataframe: raw data
       k: int: number of features we want to keep
    return: X_red: array: the k selected features
            X_filtered: pandas dataframe: datas from the selected features
    '''
    Select = SelectKBest(chi2, k=k)
    Select.fit(X, Y)
    X_red = Select.transform(X)
    selected_features = Select.get_support(indices=True)
    X_filtered = df_X.iloc[1:, selected_features]
    return X_red, X_filtered


def Heatmap(acti, auc, optimizers, learning_rates, cmap, cbarlabel = ''):
    '''Draw a heatmap for given parameters.
       acti: string: name of the activation function used to get AUCs (area under curve)
       auc: list of floats: list of AUCs got for each optimizer and for each learning rate for the given activation function
       optimizers: list of string: names of the optimizers used to get AUCs
       learning_rates: list of floats: values of the learning rates used to get AUCs
       cmap: string: color map
       cbarlabel: string: label of the color legend (by default nothing)
    Show: the heatmap of AUCs depending on the optimizers and the learning rates
    '''
    learning_rates = [round(i, 2) for i in learning_rates]

    for i in range(len(auc)):
        for j in range(7):
            auc[i][j] = round(auc[i][j], 2)
    auc = np.array(auc)

    fig, ax = plt.subplots()
    im = ax.imshow(auc)
    if not ax:
        ax = plt.gca()

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    # Axes labelling
    ax.set_xticks(np.arange(len(optimizers)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels(optimizers)
    ax.set_yticklabels(learning_rates)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    for i in range(len(learning_rates)):
        for j in range(len(optimizers)):
            text = ax.text(j, i, auc[i, j],
                           ha='center', va='center', color='black')
    ax.set_xticks(np.arange(auc.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(auc.shape[0] + 1) - .5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title('Model parameters tuning '+'Activation: '+acti+ ' epochs = 100')
    fig.tight_layout()
    plt.savefig('output/heatmap_'+acti+'_'+cbarlabel+'.png')
    plt.show()


# optimizers functions
# set the different optimizers with the given learning rate (learning_rate: int)
def Sgd(learning_rate):
    sgd = optimizers.SGD(lr = learning_rate)
    return sgd

def Rms(learning_rate):
    rms = optimizers.RMSprop(lr = learning_rate)
    return rms

def Adagrad(learning_rate):
    adagrad = optimizers.Adagrad(lr = learning_rate)
    return adagrad

def Adadelta(learning_rate):
    adadelta = optimizers.Adadelta(lr = learning_rate)
    return adadelta

def Adam(learning_rate):
    adam = optimizers.Adam(lr = learning_rate)
    return adam

def Adamax(learning_rate):
    adamax = optimizers.Adamax(lr = learning_rate)
    return adamax

def Nadam(learning_rate):
    nadam = optimizers.Nadam(lr = learning_rate)
    return nadam


def aucFromHistory(histories, score, i='None'):
    '''Get the train AUC (area under curve) and test AUC for the wanted score ('acc' or 'loss') from histories.
       histories: keras object or list of keras objects: contains the fitted model(s)
       score: string: 'acc' or 'loss' depending on the AUCs you want
       i: int: if histories contains several models, index of the model you want to get AUCs from.
               If histories contains only one model, don't fill this parameter (it will be default 'None')
    return: score_train_AUC: float: AUC for the train part
            score_test_AUC: float: AUC for the test part
    '''
    ## Get the train and test scores from histories
    # for histories containing 1 model
    if i == 'None':
        score_train = histories.history[score]
        score_test = histories.history['val_'+score]

    # for histories containing several models
    else:
        score_train = histories[i].history[score]
        score_test = histories[i].history['val_'+score]

    ## Calculate the AUC for train and for test
    score_train_AUC = metrics.auc(range(len(score_train)), score_train)
    score_test_AUC = metrics.auc(range(len(score_test)), score_test)

    return score_train_AUC, score_test_AUC


def Compare_activation(histories, activation_name, opti):
    '''Show the leaning curves for each activation function.
       histories: list of keras objects: fitted models with given optimizer and each activation function
       activation_name: list of string: names of the activation functions you want to see
       opti: string: name of the optimizer used for the models
    Show: 1 learning curve by activation function
    '''
    # Define plot
    plt.figure(figsize=(16, 16))
    mpl.style.use('seaborn')
    plt.suptitle('Metrics evolution using different activations, Optimizer = '+opti)

    # for each model (each activation function)
    for i in range(len(histories)):
        # Define the plot for this activation function
        plt.subplot(230 + i + 1)
        plt.title(activation_name[i])

        # Plot the learning curve
        plt.plot(range(100), histories[i].history['acc'], 'b')
        plt.plot(range(100), histories[i].history['val_acc'], 'r')
        plt.plot(range(100), histories[i].history['loss'], 'b--')
        plt.plot(range(100), histories[i].history['val_loss'], 'r--')

        # x axes names
        if i > 3:
            plt.xlabel('epochs')

        # Get the train and test ratios (AUC acc / AUC loss)
        auc_train_acc, auc_test_acc = aucFromHistory(histories, 'acc', i)
        auc_train_loss, auc_test_loss = aucFromHistory(histories, 'loss', i)
        train_ratio = auc_train_acc/auc_train_loss
        test_ratio = auc_test_acc/auc_test_loss

        # Print on each graph the train and test ratio
        plt.annotate('Train ratio: '+str(round(train_ratio, 2)), (0, 0.5))
        plt.annotate('Test ratio: '+str(round(test_ratio, 2)), (0, 0.4))

        # Add legend
        plt.legend(['train', 'test'], loc=7)

    # Show and save the learning curves
    plt.savefig('output/activations_curves_'+opti+'.png')
    plt.show()


def parameter_tuning(X_train, X_test, y_train, y_test, Optimizers, activations):
    '''Show test AUCs and train AUCs heatmaps for each optimizer for each given activation function
       X_train: dataframe: train data
       X_test: dataframe: test data
       y_train: dataframe: train labels
       y_test: dataframe: test labels
       Optimizers: list of string: optimizers names
       activations: list of string: activations names
    Show: run Heatmap() for train AUCs and Heatmap() for test AUCs, for each activation function
    '''
    # for each activation function
    for acti in activations:
        # Initialize both AUCs list which will contain all AUCs for this activation function
        Train_AUCs = []
        Test_AUCs = []
        # Initialize our leaning rates list which will contain all used learning rates
        learning_rates = []

        # for each learning rate
        for learning_rate in np.arange(0.01, 0.1, 0.01):
            learning_rates.append(learning_rate)
            # Get our optimizers list which will contain all our set optimizers with the appropriate learning rate
            optimizers = []
            optimizers.append(Sgd(learning_rate))
            optimizers.append(Rms(learning_rate))
            optimizers.append(Adagrad(learning_rate))
            optimizers.append(Adadelta(learning_rate))
            optimizers.append(Adam(learning_rate))
            optimizers.append(Adamax(learning_rate))
            optimizers.append(Nadam(learning_rate))

            # Initialize our train and test AUCs lists which will contain all train and test AUCs for this learning rate
            auc_list_train = []
            auc_list_test = []

            # for each optimizer
            for Optimizer in optimizers:
                # Create and fit the model with the given optimizer being set with the given learning rate and using the given activation function
                history = NN_model(X_train, X_test, y_train, y_test, Optimizer, activation=acti)

                # Get train and test accuracies AUCs
                AUC_train, AUC_test = aucFromHistory(history, 'acc')
                # add them to the lists which contain all AUCs for this learning rate
                auc_list_train.append(AUC_train)
                auc_list_test.append(AUC_test)
            # add them to AUCs list which contain all AUCs for this activation function
            Train_AUCs.append(auc_list_train)
            Test_AUCs.append(auc_list_test)

        # Show train and test heatmaps for this activation function (with all optimizers with all learning rates)
        Heatmap(acti, Train_AUCs, Optimizers, learning_rates, cmap='YlGn', cbarlabel='Training_AUC')
        Heatmap(acti, Test_AUCs, Optimizers, learning_rates, cmap='PuRd', cbarlabel='Testing_AUC')


def LearningCurve(history, title):
    '''Show the learning curve for a fitted model.
       history: keras object: fitted model
       title: string: general title
    Show: the learning curve of the model
    '''
    # set the plot
    fig = plt.figure(figsize = (16,16))
    mpl.style.use('seaborn')
    fig.suptitle(title)

    # set the accuracy learning curve
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(history.history['acc'],'b')
    ax1.plot(history.history['val_acc'],'r')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    ax1.legend(['train', 'test'], loc='lower right')

    # set the loss learning curve
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'],'b')
    ax2.plot(history.history['val_loss'],'r')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    ax2.legend(['train', 'test'], loc='upper right')

    plt.savefig('output/learning_curve.png')
    plt.show()

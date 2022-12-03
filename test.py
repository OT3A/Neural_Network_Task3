import pathlib
import csv
import cv2
import math
from random import shuffle
from calendar import month
from re import X
import pandas as pd
import numpy as np
from scipy.special import expit
from numpy import NaN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, preprocessing
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, confusion_matrix, r2_score, accuracy_score
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#================================= constants ======================================
# INPUT_LAYER = 784
# OUTPUT_LAYER = 10
INPUT_LAYER = 5
OUTPUT_LAYER = 3
#==================================================================================

def killFinite(x):
    x = np.nan_to_num(x)
    if np.any(np.isinf(x)):
        tmp = np.mean(x[np.isfinite(x)])
        s = [tmp if np.isinf(j) else j for j in x]
    
    return x


def sig(x):
    sigmoid = []
    for i in x:
        s = expit(i)
        s = killFinite(s)
        sigmoid.append(s)
    
    return np.array(sigmoid)
        

def hyperbolic_tan(x):
    x = x.reshape(-1)
    tang = [math.tanh(i) for i in x]
    return np.array(tang)


def initializeWeight(nodes):
    weights = []
    biases = []
    for i in range(len(nodes)):
        if i == 0:
            # w = np.random.randn(INPUT_LAYER, nodes[i])
            w = np.random.randn(INPUT_LAYER, nodes[i]) * np.sqrt(1/nodes[i])
            b = np.random.randn(1, nodes[i]) * np.sqrt(1/nodes[i])
        else: 
            # w = np.random.randn(nodes[i-1], nodes[i])
            w = np.random.randn(nodes[i-1], nodes[i]) * np.sqrt(1/nodes[i-1])
            b = np.random.randn(1 ,nodes[i]) * np.sqrt(1/nodes[i-1])

        weights.append(w)
        biases.append(b)

    w = np.random.randn(nodes[-1], OUTPUT_LAYER) * np.sqrt(1/nodes[-1])
    b = np.random.randn(1, OUTPUT_LAYER) * np.sqrt(1/nodes[-1])

    weights.append(w)
    biases.append(b)

    return np.array(weights), biases


def build_NN(x, y, is_bias, nodes, epochs, eta, method):
    print(f'============================= start building NN ===========================================')
    print(f'activation = {method}, eta = {eta}, epochs = {epochs}, layers = {len(nodes)}, nodes = {nodes}')
    weights, biases = initializeWeight(nodes)
    # for i in range(epochs):
        # print(f'epoch = {i}')
    # for index, row in x.iterrows():
    for i in range(len(x)):
        layers_output, predictions = forwardpropagation(x[i], y[i], weights, nodes, is_bias, biases, method)
        sigma = backpropagation(y[i], nodes, weights, layers_output)
        weights = update_weights(sigma, weights, biases, eta, x[i], layers_output, nodes)
        if i == epochs:
            break
    print(f'============================= NN is ready to use ===========================================')
    return weights, biases


def update_weights(sigma, weights, biases, eta, x, output, nodes):
    for i in range(len(nodes) + 1):
        s = np.array(sigma[i])
        s = killFinite(s)
        if i == 0:
            delta = []
            for j in s:
                delta.append(eta * j * x)
            delta = killFinite(delta)
            delta = np.array(delta)
            weights[i] = weights[i] + delta.T
            # biases[i] = biases[i] + delta.T
        else:
            delta = []
            for j in range(len(s)):
                delta.append(eta * s[j] * output[i][j])
            delta = killFinite(delta)
            delta = np.array(delta)
            weights[i] = weights[i] + delta.T
            # biases[i] = biases[i] + delta.T
        
        weights[i] = killFinite(weights[i])

    return weights


def backpropagation(y, nodes, weights, layers_output):
    sigmas = []
    for i in range(len(nodes), -1, -1):
        sigma = []
        if i == len(nodes):
            for j in range(OUTPUT_LAYER):
                hidden_sigma = (layers_output[i][j] - y[j]) * (layers_output[i][j]) * (1 - layers_output[i][j])
                sigma.append(hidden_sigma)
        else:
            for j in range(nodes[i]):
                # e = sum(last_sigma * weights[i+1][j])
                e = np.dot(last_sigma, weights[i+1][j])
                hidden_sigma = e * (layers_output[i][j]) * (1 - layers_output[i][j])
                sigma.append(hidden_sigma)
        sigma = killFinite(sigma)
        last_sigma = sigma
        sigmas.insert(0, last_sigma)

    return np.array(sigmas)


def forwardpropagation(x, y, weights, nodes, is_bias, biases, method):
    input = None
    output = []
    prediction = []
    for i in range(len(nodes) + 1):
        if is_bias:
            if i == 0:
                net = weights[i].T.dot(x) + biases[i]
            else:
                net = weights[i].T.dot(input) + biases[i]   
        else:
            if i == 0:
                net = weights[i].T.dot(x)
            else:
                net = weights[i].T.dot(input)
        net = killFinite(net)
        if method == 'sigmoid':
            acc = sig(net)
        else: 
            acc = hyperbolic_tan(net)
        acc = np.array(acc)
        acc = acc.reshape(-1)
        acc = killFinite(acc)
        input = acc
        output.append(acc)
        if i == len(nodes):
            prediction.append(acc)

    yhat = np.argmax(prediction)
    prediction = np.zeros(OUTPUT_LAYER)
    prediction[yhat] = 1
    return output, prediction


def test(x, y, weights, nodes, is_bias, biases, method):
    predictions = []
    # for index, row in x.iterrows():
    for i in range(len(x)):
        _, p = forwardpropagation(x[i], y[i], weights, nodes, is_bias, biases, method)
        predictions.append(p)
    predictions = np.array(predictions)
    accuracy = accuracy_score(y, predictions)
   
    # print(f'y shape = {y.shape}, prediction shape = {predictions.shape}')
    # cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(predictions).argmax(axis=1))
    # plt.figure(figsize = (10,8))
    # sns.heatmap(cm, annot=True, cmap= 'flare',  fmt='d', cbar=True)
    # plt.xlabel('Predicted_Label')
    # plt.ylabel('Truth_Label')
    # plt.title('Confusion Matrix')
    # plt.show()
    return accuracy


def labelEncoder(Y):
    encoded_y = []
    for i in range(Y.shape[0]):
        y = np.zeros(OUTPUT_LAYER)
        y[Y[i]] = 1
        encoded_y.append(y)
    
    return np.array(encoded_y)

def penguins_encoding(Y):
    encoded_y = []
    for i in Y:
        if i == 'Adelie':
            encoded_y.append(0)
        elif i == 'Gentoo':
            encoded_y.append(1)
        else:
            encoded_y.append(2)
    
    return np.array(encoded_y)

def read_numbers():
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')
    normalizer = preprocessing.MinMaxScaler()
    
    x_train = train_data.iloc[:, 1:]
    x_train = normalizer.fit_transform(x_train)
    
    x_test = test_data.iloc[:, 1:]
    x_test = normalizer.fit_transform(x_test)
    
    y_train = train_data.loc[:, 'label']
    y_train = labelEncoder(y_train)
    
    y_test = test_data.loc[:, 'label']
    y_test = labelEncoder(y_test)
    
    return x_train, y_train, x_test, y_test

def read_penguins():
    data = pd.read_csv('penguins.csv')
    data = data.sample(frac = 1)
    y_train = data.loc[:, 'species']
    x_train = data.iloc[:, 1:]
    y_test = data.loc[:, 'species']
    x_test = data.iloc[:, 1:]

    normalizer = preprocessing.StandardScaler()
    x_train['gender'] = x_train['gender'].replace(['male', 'female', NaN], [0, 1, 0])
    x_train['gender'] = x_train['gender'].astype('int')
    x_train = normalizer.fit_transform(x_train)
    # print(x_train)
    x_test['gender'] = x_test['gender'].replace(['male', 'female', NaN], [0, 1, 0])
    x_test['gender'] = x_test['gender'].astype('int')
    x_test = normalizer.fit_transform(x_test)
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.10, shuffle=True)
    
    # x_train = x_train.reset_index()
    # x_test = x_test.reset_index()
    # y_train = y_train.reset_index()
    # y_test = y_test.reset_index()
    
    y_train = penguins_encoding(y_train)
    y_test = penguins_encoding(y_test)
    y_train = labelEncoder(y_train)
    y_test = labelEncoder(y_test)
   
    
    return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = read_numbers()
x_train, y_train, x_test, y_test = read_penguins()

def main(activation_func, nodes, eta, epochs, is_bias):
    weights, biases = build_NN(x_train, y_train, is_bias, nodes, epochs, eta, activation_func)
    print(f'========================================== start testing ===================================================')
    train_accuracy = test(x_train, y_train, weights, nodes, is_bias, biases, activation_func)
    test_accuracy = test(x_test, y_test, weights, nodes, is_bias, biases, activation_func)
    print(f'train accuracy = {train_accuracy}, test accuracy = {test_accuracy}')


def run(activation_func, nodes, eta, epochs, is_bias):
    weights, biases = build_NN(x_train, y_train, is_bias, nodes, epochs, eta, activation_func)
    print(f'========================================== start testing ===================================================')
    train_accuracy = test(x_train, y_train, weights, nodes, is_bias, biases, activation_func)
    test_accuracy = test(x_test, y_test, weights, nodes, is_bias, biases, activation_func)
    print(f'train accuracy = {train_accuracy}, test accuracy = {test_accuracy}')
    return test_accuracy


if __name__ == '__main__':
    nodes = [3,4]
    main('sigmoid', nodes, 0.1, 10000, False)
    # main('tanh', nodes, 0.1, 1000, False)
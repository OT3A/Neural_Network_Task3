import pathlib
import csv
import cv2
import math
from random import shuffle
from calendar import month
from re import X
import pandas as pd
import numpy as np
from numpy import NaN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
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
def read():
    train_data = pd.read_csv('~/Downloads/archive/mnist_train.csv')
    test_data = pd.read_csv('~/Downloads/archive/mnist_test.csv')
    y_train = train_data.loc[:, 'label']
    x_train = train_data.iloc[:, 1:]
    y_test = test_data.loc[:, 'label']
    x_test = test_data.iloc[:, 1:]
    # if is_bias == True:
    #     bias = pd.Series(np.ones(x_train.shape[0]))
    # else:
    #     bias = pd.Series(np.zeros(x_train.shape[0]))
    
    # x_train = pd.concat([bias, x_train], axis=1)
    y_train = labelEncoder(y_train)
    y_test = labelEncoder(y_test)
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = read()


def sig(x):
    sigmoid = [1 / 1 + np.float128(np.exp(-i)) for i in x]
    # return sigmoid
    return np.array(sigmoid).astype(np.float)
    # sigmoid = []
    # for i in x:
    #     if -i > np.log(np.finfo(type(i)).max):
    #         sigmoid.append(0.0)    
    #     else:
    #         sigmoid.append(1 / 1 + np.exp(-i))
    
    # # return sigmoid
    # return np.array(sigmoid)
        

def hyperbolic_tan(x):
    tang = [math.tanh(i) for i in x]
    return tang


def initializeWeight(nodes, is_bias):
    weights = []
    for i in range(len(nodes)):
        if is_bias:
            if i == 0:
                # w = np.random.randn(784, nodes[i])
                w = np.random.randn(784 + nodes[i], nodes[i])
                # b = np.random.randn(784, nodes[i])
            else: 
                # w = np.random.randn(nodes[i-1], nodes[i])
                w = np.random.randn(nodes[i-1] + nodes[i], nodes[i])
                # b = np.random.randn(nodes[i-1] ,nodes[i])
        else:
            if i == 0:
                # w = np.random.randn(784, nodes[i])
                w = np.random.randn(784, nodes[i])
                # b = np.random.randn(784, nodes[i])
            else: 
                # w = np.random.randn(nodes[i-1], nodes[i])
                w = np.random.randn(nodes[i-1], nodes[i])
                # b = np.random.randn(nodes[i-1] ,nodes[i])

        # calculate the range for the weights
        lower, upper = -(math.sqrt(6.0) / math.sqrt(nodes[i-1] + nodes[i])), (math.sqrt(6.0) / math.sqrt(nodes[i-1] + nodes[i]))
        # scale to the desired range
        w = lower + w * (upper - lower)
        weights.append(w)

    w = np.random.randn(nodes[-1] + 10, 10) if is_bias == True else np.random.randn(nodes[-1], 10)
    # calculate the range for the weights
    lower, upper = -(math.sqrt(6.0) / math.sqrt(nodes[-1] + 10)), (math.sqrt(6.0) / math.sqrt(nodes[-1] + 10))
    # scale to the desired range
    w = lower + w * (upper - lower)
    weights.append(w)

    return np.array(weights)


def build_NN(x, y, is_bias, nodes, epochs = 2, eta = 0.01):
    print(f'============================= start building NN ===========================================')
    weights = initializeWeight(nodes, is_bias)
    # print(f'weights = {weights[0].shape}, {weights[1].shape}, {weights[2].shape}')
    for i in range(epochs):
        print(f'epoch = {i}')
        for index, row in x.iterrows():
            layers_output, predictions = forwardpropagation(row, y[index], weights, nodes, is_bias)
            sigma = backpropagation(y[index], nodes, weights, layers_output)
            weights = update_weights(sigma, weights, eta, row, layers_output, nodes)

    print(f'============================= NN is ready to use ===========================================')
    return weights


def update_weights(sigma, weights, eta, x, output, nodes):
    # weights = weights + eta * sigma * input
    # s = len(nodes)
    for i in range(len(nodes) + 1):
        if i == 0:
            # print(f'sigma = {sigma[i]}')
            # print(f'weight = {weights[i].shape}')
            delta = []
            s = np.array(sigma[i])
            for j in s:
                # for k in j:
                delta.append(eta * j * x)
            delta = np.array(delta)
            weights[i] = weights[i] + delta.T
            # weights[i] = weights[i] + eta * np.array(sigma[i]) * x
        else:
            delta = []
            s = np.array(sigma[i])
            for j in range(len(s)):
                # for k in j:
                delta.append(eta * s[j] * output[i][j])
            delta = np.array(delta)
            weights[i] = weights[i] + delta.T
            # weights[i] = weights[i] + eta * sigma[i] * output[i-1]

    return weights


def backpropagation(y, nodes, weights, layers_output):
    sigmas = []
    for i in range(len(nodes), -1, -1):
        sigma = []
        if i == len(nodes):
            for j in range(10):
                hidden_sigma = (layers_output[i][j] - y[j]) * (layers_output[i][j]) * (1 - layers_output[i][j])
                sigma.append(hidden_sigma)
        else:
            # sigma = (last_sigma * weights[i]) * (layers_output[i]) * (1 - layers_output[i])
            # for s in range(len(last_sigma)):
            #     temp = []
            #     for j in range(nodes[i]):
            #         hidden_sigma = (last_sigma[s] * weights[i][j]) * (layers_output[i][j]) * (1 - layers_output[i][j])
            #         temp.append(hidden_sigma)
            #     sigma.append(temp)
            for j in range(nodes[i]):
                # for s in range(len(last_sigma)):
                # print(f'{i}, {j}, {weights[i][j]}')
                hidden_sigma = sum(last_sigma * weights[i+1][j]) * (layers_output[i][j]) * (1 - layers_output[i][j])
                sigma.append(hidden_sigma)

        last_sigma = sigma
        sigmas.insert(0, last_sigma)
        # sigmas = np.insert(sigmas, 0, last_sigma)

    return np.array(sigmas)


def forwardpropagation(x, y, weights, nodes, is_bias, method='sigmoid'):
    input = None
    output = []
    prediction = []
    for i in range(len(nodes) + 1):
        if is_bias:
            if i == 0:
                net = weights[i][nodes[i]:].T.dot(x) + weights[i][0]
            else:
                net = weights[i][nodes[i]:].T.dot(input) + weights[i][0]
        else:
            if i == 0:
                net = weights[i].T.dot(x)
            else:
                net = weights[i].T.dot(input)

        if method == 'sigmoid':
            acc = sig(net)
        else: 
            acc = hyperbolic_tan(net)
        input = acc
        output.append(acc)
        if i == len(nodes):
            prediction.append(acc)

    yhat = np.argmax(prediction)
    prediction = np.zeros(10)
    prediction[yhat] = 1
    return output, prediction


def test(x, y, weights, nodes, is_bias):
    print(f'========================================== start testing ===================================================')
    predictions = []
    for index, row in x.iterrows():
        _, p = forwardpropagation(row, y[index], weights, nodes, is_bias)
        predictions.append(p)
        # print(f'p = {p}')
    predictions = np.array(predictions)
    # print(f'y = {y[10000-1]},prediction = {(predictions[10000-1])}')
    # print(f'y shape = {y.shape}, prediction shape= {predictions.shape}')
    # print(f'r2 accuracy = {r2_score(y, predictions)}')
    print(f'accuracy = {accuracy_score(y, predictions)}')
    print(f'========================================== end testing ===================================================')


def labelEncoder(Y):
    encoded_y = []
    for i in range(Y.shape[0]):
        y = np.zeros(10)
        y[Y[i]] = 1
        encoded_y.append(y)
    
    return np.array(encoded_y)


def main(is_bias = False):
    # train_data = pd.read_csv('~/Downloads/archive/mnist_train.csv')
    # test_data = pd.read_csv('~/Downloads/archive/mnist_test.csv')
    # y_train = train_data.loc[:, 'label']
    # x_train = train_data.iloc[:, 1:]
    # y_test = test_data.loc[:, 'label']
    # x_test = test_data.iloc[:, 1:]
    # if is_bias == True:
    #     bias = pd.Series(np.ones(x_train.shape[0]))
    # else:
    #     bias = pd.Series(np.zeros(x_train.shape[0]))
    
    # # x_train = pd.concat([bias, x_train], axis=1)
    # y_train = labelEncoder(y_train)
    # y_test = labelEncoder(y_test)
    nodes = [2, 2]
    weights = build_NN(x_train, y_train, is_bias, nodes)
    test(x_test, y_test, weights, nodes, is_bias)


if __name__ == '__main__':
    main()
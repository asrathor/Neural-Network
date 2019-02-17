'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
import numpy as np
from scipy.optimize import minimize


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    """
    FeedForward pass
    biasMatrix: matrix containing one column of bias (1)
    """

    biasMatrix = np.ones((len(training_label)))
    training_data = np.c_[training_data, biasMatrix]

    aj = training_data.dot(w1.T)  # Equation 1
    zj = sigmoid(aj)  # Equation 2
    zj_bias = np.c_[zj, np.ones(zj.shape[0])]

    bl = zj_bias.dot(w2.T)  # Equation 3
    ol = sigmoid(bl)  # Equation 4

    """
    # 1 to k initialization of training label to convert it to vector
    # A matrix of 0s is added to labels to make the total columns as 2. Then the value in first column is converted
      to 0 and appropriate index equal to initial value is made 1.
    """
    training_label = np.c_[training_label, np.zeros((len(training_label), 1))]
    yl = np.zeros(training_label.shape)

    for i in range(training_label.shape[0]):
        temp = training_label[i, 0]
        yl[i, int(temp)] = 1.0

    """
    # Equation 8 and 9 are implemented to calculate the error (named errorOut) and delta.
    """
    delta = ol - yl  # Equation 9
    errorOut = delta.T.dot(zj_bias)  # Equation 8

    """
    # Equation 12 is used to compute the second error (named errorHidden)
    """
    delta = delta.dot(w2)  # Summation part of Equation 12
    y_val = (1 - zj_bias) * zj_bias
    delta_Y = delta * (y_val)
    errorHidden = np.dot(delta_Y.T, training_data)  # Equation 12

    """
    # Error calculated through Equation 5 which is negative log-likelihood error function
    """
    ln_ol = np.log(1 - ol)
    ln_yo = np.multiply((1 - yl), ln_ol)
    ln_yl = np.multiply(yl, np.log(ol))
    lnoy = np.add(ln_yo, ln_yl)
    lnoy = np.sum(lnoy, 1)
    lnoy = np.sum(lnoy)
    lnoy = np.negative(lnoy) / len(training_data)  # Equation 5

    """
    # Regularization for obj_val
    """
    sqw1 = np.square(w1)
    sumw1_one = np.sum(sqw1, 1)
    sumw1_two = np.sum(sumw1_one)
    sqw2 = np.square(w2)
    sumw2_one = np.sum(sqw2, 1)
    sumw2_two = np.sum(sumw2_one)
    sum_total = sumw1_two + sumw2_two
    l_n = lambdaval / (2 * len(training_data))
    result = l_n * sum_total  # Equation 15

    obj_val = lnoy + result

    print("Error func: ", obj_val)

    """
    # ErrorOut subject to regularization
    """
    quan_one = np.multiply(lambdaval, w2)
    errorOut = (quan_one + errorOut) / len(training_data)  # Equation 16

    """
    # ErrorHidden subject to regularization
    """
    quan_oneH = np.multiply(lambdaval, w1)
    errorHidden = np.delete(errorHidden, errorHidden.shape[0] - 1, 0)
    errorHidden = (quan_oneH + errorHidden) / len(training_data)  # Equation 17

    """
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices
    are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    """

    obj_grad = np.concatenate((errorHidden.flatten(), errorOut.flatten()), 0)
    # obj_grad = obj_grad/len(training_data)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """
    # For Prediction same steps are followed as in FeedForward pass but instead of training_data, data from argument
      is used.
    """
    biasMatrix = np.ones((len(data)))
    data = np.c_[data, biasMatrix]

    hidden_out = data.dot(w1.T)
    hidden_out = sigmoid(hidden_out)
    hidden_bias = np.c_[hidden_out, np.ones(hidden_out.shape[0])]

    output = hidden_bias.dot(w2.T)
    output_sig = sigmoid(output)
    outputs = np.argmax(output_sig, axis=1)
    labels = outputs

    return labels


def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
lambdaval = 35;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
params = nn_params.get('x')
# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data)
# find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, validation_data)
# find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
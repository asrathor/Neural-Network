import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """
    # Sigmoid function is used as an activation function
    """
    return 1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    """
    # Feature selection is done to remove values which are exactly the same for all data points in training data
    # check_values is a vector which contains boolean values depicting whether the respective column in training
      data contains same or different values. True if it contains the same value.
    # The feature selection is implemented on training, validation and test data.
    """
    check_values = np.all(train_data == train_data[0, :], axis=0)
    pointer = 0
    global selected_features
    selected_features = []
    for i, j in enumerate(check_values):
        if j == False:
            train_data[:, pointer] = train_data[:, i]
            test_data[:, pointer] = test_data[:, i]
            validation_data[:, pointer] = validation_data[:, i]
            selected_features = np.append(selected_features, i)
            pointer = pointer + 1
    train_data = train_data[:50000, :pointer]
    test_data = test_data[:10000, :pointer]
    validation_data = validation_data[:10000, :pointer]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

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

    aj = training_data.dot(w1.T)                # Equation 1
    zj = sigmoid(aj)                            # Equation 2
    zj_bias = np.c_[zj, np.ones(zj.shape[0])]

    bl = zj_bias.dot(w2.T)                      # Equation 3
    ol = sigmoid(bl)                            # Equation 4

    """
    # 1 to k initialization of training label to convert it to vector
    # A matrix of 0s is added to labels to make the total columns as 10. Then the value in first column is converted
      to 0 and appropriate index equal to initial value is made 1.
    """
    training_label = np.c_[training_label, np.zeros((len(training_label), 9))]
    yl = np.zeros(training_label.shape)

    for i in range(training_label.shape[0]):
        temp = training_label[i, 0]
        yl[i, int(temp)] = 1.0

    """
    # Equation 8 and 9 are implemented to calculate the error (named errorOut) and delta.
    """
    delta = ol-yl                         # Equation 9
    errorOut = delta.T.dot(zj_bias)       # Equation 8

    """
    # Equation 12 is used to compute the second error (named errorHidden)
    """
    delta = delta.dot(w2)                 # Summation part of Equation 12
    y_val = (1 - zj_bias) * zj_bias
    delta_Y = delta * (y_val)
    errorHidden = np.dot(delta_Y.T, training_data)    # Equation 12

    """
    # Error calculated through Equation 5 which is negative log-likelihood error function
    """
    ln_ol = np.log(1 - ol)
    ln_yo = np.multiply((1 - yl), ln_ol)
    ln_yl = np.multiply(yl, np.log(ol))
    lnoy = np.add(ln_yo, ln_yl)
    lnoy = np.sum(lnoy, 1)
    lnoy = np.sum(lnoy)
    lnoy = np.negative(lnoy) / len(training_data)     # Equation 5

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
    result = l_n * sum_total                         # Equation 15

    obj_val = lnoy + result

    print("Error func: ", obj_val)

    """
    # ErrorOut subject to regularization
    """
    quan_one = np.multiply(lambdaval, w2)
    errorOut = (quan_one + errorOut) / len(training_data)       # Equation 16

    """
    # ErrorHidden subject to regularization
    """
    quan_oneH = np.multiply(lambdaval, w1)
    errorHidden = np.delete(errorHidden, errorHidden.shape[0] - 1, 0)
    errorHidden = (quan_oneH + errorHidden) / len(training_data)    # Equation 17

    """
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices
      are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    """

    obj_grad = np.concatenate((errorHidden.flatten(), errorOut.flatten()), 0)
    #obj_grad = obj_grad/len(training_data)

    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image
    % Output:
    % label: a column vector of predicted labels"""

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
    outputs = np.argmax(output_sig,axis=1)
    labels = outputs

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
print("iniW size: ", initialWeights.shape)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

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

# Pickle
obj = [selected_features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj, open('params.pickle', 'wb'))
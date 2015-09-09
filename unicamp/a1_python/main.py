# -*- coding: utf-8 -*-

import numpy as numpy
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def normalize_features(array):
    """
    Normalize the features in the data set.
    """
    mu = array.mean()
    sigma = array.std()
    array_normalized = (array - mu) / sigma

    return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iters):
    m = len(values)
    cost_history = []

    for i in range(num_iters):
        print "Current iteration: %d " % i
        predicted_values = numpy.dot(features, theta)
        theta = theta - alpha / m * numpy.dot((predicted_values - values), features)

        cost = compute_cost(features, values, theta)
        cost_history.append(cost)

    return theta, cost_history

def plot_cost_history(alpha, cost_history):
    """This function is for viewing the plot of your cost history.
    You can run it by uncommenting this
        plot_cost_history(alpha, cost_history)
    call in predictions.
    If you want to run this locally, you should print the return value
    from this function.
    """
    cost_df = pd.DataFrame({
        'Cost_Function': cost_history,
        'Iterations': range(len(cost_history))
    })

    return ggplot(cost_df, aes('Iterations', 'Cost_Function')) + geom_point() + ggtitle(
        'Cost Function - alpha = %.3f' % alpha)

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced
    # predictions.
    #
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    d = numpy.array(data)
    p = numpy.array(predictions)
    m = numpy.mean(d)

    r_squared = 1 - numpy.square(d - p).sum() / numpy.square(d - m).sum()

    return r_squared

def plot_residuals(data, predictions):
    """
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:
    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    """

    plt.figure()
    (data - predictions).hist()

    return plt

def gd_method(f, f_valid, v, v_validation):
    m = len(v)
    m_validation = len(v_validation)

    # Normalize data
    features, mu, sigma = normalize_features(f)

    # Normalize validation data (using the mean and std calculated when normalizing the model)
    features_validation, mu, sigma = normalize_features(f_valid)

    features['ones'] = numpy.ones(m)  # Add a column of 1s (y intercept)
    features_validation['ones'] = numpy.ones(m_validation)  # Add a column of 1s (y intercept)

    # Convert features and values to numpy arrays
    features_array = numpy.array(features)
    values_array = numpy.array(v)

    # Set values for alpha, number of iterations.
    alpha = 0.04  # please feel free to change this value
    num_iterations = 1000  # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = numpy.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array,
                                                            values_array,
                                                            theta_gradient_descent,
                                                            alpha,
                                                            num_iterations)
    plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    plot = plot_cost_history(alpha, cost_history)
    print plot
    #
    # Please note, there is a possibility that plotting
    # this in addition to your calculation will exceed
    # the 30 second limit on the compute servers.

    # Predictions for the data used for define the model
    predictions = numpy.dot(features, theta_gradient_descent)
    print predictions

    # Predictions for the validation data
    pred_validation = numpy.dot(features_validation, theta_gradient_descent)
    print pred_validation

    # Compute error using mean absolute error
    print "MEAN ABSOLUTE ERROR "
    print mean_absolute_error(v, predictions)

    print "MEAN ABSOLUTE ERROR (validation) "
    print mean_absolute_error(v_validation, pred_validation)


def normalEquation(features, features_validation, values, values_validation):

    M = numpy.dot(features.T, features)
    print "Transposta de f por f"
    print M.shape
    M = numpy.array(M)
    print "Transformou em array"
    print M.shape
    M = numpy.linalg.pinv(M)
    print "Inversa"
    print M.shape
    M = numpy.dot(M, features.T)
    print "Multiplicou por transposta de f"
    print M.shape
    theta = numpy.dot(M, values)
    #M = numpy.linalg.pinv(M)


    print theta.shape
    print features.shape
    print theta

    predictions = numpy.dot(theta, features.T)
    pred_validation = numpy.dot(theta, features_validation.T)

    print predictions

    print "MEAN ABSOLUTE ERROR "
    print mean_absolute_error(values, predictions)

    print "MEAN ABSOLUTE ERROR (validation) "
    print mean_absolute_error(values_validation, pred_validation)


if __name__ == '__main__':
    # Read data
    data = pd.read_csv('C:\\Users\\Victor Leal\\Desktop\\mlearning\\assignment1\\YearPredictionMSD.txt', header=None)
    #data = pd.read_csv('/home/victor/YearPredictionMSD.txt', header=None)

    '''
    515345 Linhas no total
    Training = 463715
    Testing = 51630

    Dividindo o training para validation
    70% do conjunto de training = 324600
    30% para validation = 139115
    '''

    # Used to define the model
    features = data.ix[range(0, 324600), range(1,91,3)]
    values = data.ix[range(0, 324600), 0]
    features_validation = data.ix[range(324600, 463715), range(1,91,3)]
    values_validation = data.ix[range(324600, 463715), 0]
    '''
    features_a = data.ix[range(0, 1000), range(1,31)]
    features_b = data.ix[range(0, 1000), range(31,61)]
    features_c = data.ix[range(0, 1000), range(61,91)]

    values = data.ix[range(0, 1000), 0]

    # Used for validate the model
    features_validation = data.ix[range(324600, 463715), range(1,91)]
    values_validation = data.ix[range(324600, 463715), 0]

    m = len(values)
    m_validation = len(values_validation)

    ones = numpy.ones(m)  # Add a column of 1s (y intercept)
    ones_validation = numpy.ones(m_validation)  # Add a column of 1s (y intercept)


    #features = numpy.append(features, ones)
    a = numpy.array(features_a).sum(axis=1)
    b = numpy.array(features_b).sum(axis=1)
    c = numpy.array(features_c).sum(axis=1)

    features = [ones,a,b,c]

    features_validation['ones'] = numpy.ones(m_validation)  # Add a column of 1s (y intercept)
    #features_validation = numpy.append(features_validation, ones_validation)

    features_array = numpy.array(features)
    features_validation_array = numpy.array(features_validation)
    values_array = numpy.array(values)
    values_validation_array = numpy.array(values_validation)
    '''
    gd_method(features, features_validation, values, values_validation)

    #normalEquation(features_array, features_validation_array, values_array, values_validation_array)



'''
    p = plot_residuals(values, predictions)
    p.show()

    p2 = plot_residuals(values_validation, pred_validation)
    p2.show()
'''
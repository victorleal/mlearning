import numpy as numpy
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

__author__ = 'Victor Leal'

if __name__ == '__main__':
    # Read data
    data = pd.read_csv('C:\\Users\\Victor Leal\\Desktop\\mlearning\\assignment1\\YearPredictionMSD.txt', header=None)

    # Used to define the model
    features = data.ix[range(0, 324600), range(1,91,3)]
    values = data.ix[range(0, 324600), 0]




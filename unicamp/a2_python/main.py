# -*- coding: utf-8 -*-

import numpy as numpy
import pandas as pd
import re
import collections
from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pylab import scatter, show, legend, xlabel, ylabel



if __name__ == '__main__':
    # Read data

    '''
    data = pd.read_csv('/home/victor/2015s2-mo444-assignment-02.csv')

    words = re.findall('\w+', ' '.join(data['Descript']).lower())
    c = collections.Counter(words)
    frequencies = c.values()

    target = data['Category']
'''
    data = numpy.loadtxt('/home/victor/2015s2-mo444-assignment-02.csv', delimiter=',')
    X = data.ix[:, 2:]
    y = data.ix[range(0,1000), 1]

    pos = numpy.where(y == 1)
    neg = numpy.where(y == 0)
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Not Admitted', 'Admitted'])
    show()




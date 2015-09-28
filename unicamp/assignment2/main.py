# -*- coding: utf-8 -*-

import time
import numpy as numpy
import pandas as pd
import re
import collections
import statsmodels.api as sm
#from ggplot import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



if __name__ == '__main__':
    # Read data
    df = pd.read_csv('/home/victor/2015s2-mo444-assignment-02.csv')
    #df = pd.read_csv('C:\\Users\\Victor Leal\\Desktop\\2015s2-mo444-assignment-02.csv')

    df['Dates'] = df['Dates'].apply(lambda d: time.strptime(d, "%Y-%m-%d %H:%M:%S"))
    df['year'] = df['Dates'].apply(lambda d: d.tm_year)
    df['month'] = df['Dates'].apply(lambda d: d.tm_mon)
    df['day'] = df['Dates'].apply(lambda d: d.tm_mday)
    df['hour'] = df['Dates'].apply(lambda d: d.tm_hour)
    df['minute'] = df['Dates'].apply(lambda d: d.tm_min)
    df['wday'] = df['Dates'].apply(lambda d: d.tm_wday)

    #print df.head()

    #cols_to_keep = ['Category', 'X', 'Y', 'Dates']
    dummy_ranks = pd.get_dummies(df['PdDistrict'], prefix='PdDistrict')

    df = df.drop(['Dates', 'DayOfWeek', 'Resolution', 'Address', 'Descript', 'PdDistrict'], axis=1)
    df = df.join(dummy_ranks.ix[:, :])

    # Just 10 thousand training examples
    df = df.ix[range(0, 10000), :]
    df['Interceptor'] = 1.0

    train_cols = df.columns[1:]
    print train_cols

    model = LogisticRegression()
    model.fit(df[train_cols], df['Category'])
    print(model)

    # make predictions
    expected = numpy.array(df['Category']).astype(str)
    predicted = model.predict(df[train_cols])
    print expected
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))

    #logit = sm.MNLogit(df['Category'], df[train_cols])
    #print logit

    # fit the model
    #result = logit.fit(method="bfgs")

    #print result.summary()

    '''

    words = re.findall('\w+', ' '.join(df['Descript']).lower())
    c = collections.Counter(words)

    frequencies = c.values()
    #print frequencies

    #print pd.crosstab(data['Category'], data['Descript'], rownames=['Category'])

    df = df.ix[range(0, 10000), :]

    dummy_ranks = pd.get_dummies(df['Descript'], prefix='Descript')

    cols_to_keep = ['Category', 'X', 'Y', 'Dates']
    data = df[cols_to_keep].join(dummy_ranks.ix[:, 'Descript_2':])


    train_cols = data.columns[1:]
    # Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

    logit = sm.Logit(data['Category'], data[train_cols])

    # fit the model
    result = logit.fit()

    print result.summary()
    '''

'''
    for i in range(1, len(frequencies), 9):
        index = "frequency_" + str(i)
        df[index] = frequencies[i]
'''
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


print "\n\n\n"
print "===== PREPARING KMEANS WITH K = 1000 MOTHERFUCKERS ====="
print "\n\n\n"

features = pd.read_csv("/home/victor/desktop/features2.csv")
#print features.ix[[1,2,3],:]

random_feat = np.random.randint(0,81432,10000)

features = features.ix[random_feat, :]
k = 1000

print "FEATURES SHAPE:"
print features.shape
print "\n\n\n"

print "===== KMEANS IS GONNA RUN BITCHES ====="
print "\n\n\n"

print "==========="
print "K:"
print k

km = KMeans(k)
km.fit(features)

print "KMEANS RESULT:"

print "CLUSTER CENTERS:"
print km.cluster_centers_

print "LENGTH OF CLUSTER CENTERS:"
print len(km.cluster_centers_)

print "INERTIA:"
print km.inertia_

print "LABELS:"
print km.labels_

print "\n\n"


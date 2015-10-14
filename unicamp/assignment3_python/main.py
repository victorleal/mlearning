# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import simplejson

# for each image in the directory, find its histogram
mypath = "/home/victor/2015s2-mo444-assignment-03/"
images = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

cont = 1
colors = ("b", "g", "r")

fv = open("/home/victor/Desktop/features.csv", "w")

for i in images:
    print cont
    # load the image and show it
    image = cv2.imread(join(mypath, i))

    channels = cv2.split(image)

    features = []

    # loop over the image channels
    for (chan, color) in zip(channels, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

    fv.write(",".join(str(e) for e in features))
    fv.write("\n")

    image = None
    cont+=1

fv.close()
# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print "flattened feature vector size: %d" % (np.array(features).flatten().shape)

'''
km = KMeans(3)
km.fit(features)

print km.cluster_centers_
print km.inertia_
print km.labels_
'''

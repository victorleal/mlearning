# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


# for each image in the directory, find its histogram
#mypath = "/home/victor/2015s2-mo444-assignment-03/"
mypath = "/home/victor/desktop/a3-images/"
images = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

cont = 1
colors = ("b", "g", "r")

fv = open("/home/victor/desktop/features2.csv", "w")

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

    fv.write(",".join(str(e[0]) for e in features))
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
#print "flattened feature vector size: %d" % (np.array(features).flatten().shape)

print "\n\n\n"
print "===== PREPARING KMEANS MOTHERFUCKERS ====="
print "\n\n\n"

features = pd.read_csv("/home/victor/desktop/features2.csv")
#print features.ix[[1,2,3],:]

random_feat = np.random.randint(0,81432,10000)

features = features.ix[random_feat, :]

print "FEATURES SHAPE:"
print features.shape
print "\n\n\n"

print "===== KMEANS IS GONNA RUN BITCHES ====="
print "\n\n\n"

for k in range(1000, 3000, 500):
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


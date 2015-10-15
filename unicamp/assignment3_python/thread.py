from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import threading

#print "flattened feature vector size: %d" % (np.array(features).flatten().shape)

# Our thread class:
class KmeansThread ( threading.Thread ):
    def __init__ ( self, k, features ):
        self.k = k
        self.features = features
        threading.Thread.__init__ ( self )

    def run ( self ):
        #print "==========="
        #print "K: %d" % self.k

        km = KMeans(self.k)
        km.fit(self.features)

        #print "KMEANS RESULT (%d):" % self.k

        #print "CLUSTER CENTERS:"
        #print km.cluster_centers_

        print "LENGTH OF CLUSTER CENTERS (K=%d): %d" % (self.k, len(km.cluster_centers_))
	
        print "INERTIA (K=%d): %f" % (self.k, km.inertia_)

        #print "LABELS:"
        #print km.labels_

        print "\n\n"
	    

print "\n\n\n"
print "===== PREPARING KMEANS MOTHERFUCKERS ====="
print "\n\n\n"

features = pd.read_csv("/home/victor/desktop/features2.csv")
#print features.ix[[1,2,3],:]

random_feat = np.random.randint(0, 81432, 10000)

features = features.ix[random_feat, :]

print "FEATURES SHAPE:"
print features.shape
print "\n\n\n"



print "===== KMEANS IS GONNA RUN BITCHES ====="
print "\n\n\n"

for k in range(1000, 3000, 500):
    print "==========="
    print "K: %d" % k
    KmeansThread(k, features).start()	


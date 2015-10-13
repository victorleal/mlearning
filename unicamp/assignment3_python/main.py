__author__ = 'victor'

import pandas as pd
from sklearn.cluster import KMeans


data = pd.read_csv("/home/victor/Desktop/mlearning/unicamp/assignment3/summarized_bic.txt")

kmeans = KMeans(300)

kmeans.fit(data)

print kmeans.cluster_centers_
print kmeans.inertia_
print kmeans.labels_
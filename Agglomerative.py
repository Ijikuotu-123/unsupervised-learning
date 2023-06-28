""" Hierarchiacal clustering is an unsupervised machine learning method for clustering data points. it builds
clusters by measuring the dissimilarities between data. Unsupervised learning means that a model does not 
have to be traine, and we do not need a "target" variable. clusters are visulaize using a dendrogram

Each data point is treated as a cluster point. then, we join clusters together that have shortest distance
between them to create larger cluster. this step is repeated until one large cluster is formed containing
all the data points.

Hierarchiacal clustering requires us to decide on both a distance and a linkage method.
Distance:
1. Euclidean distance 
2. square euclidean distance
3. Manhattan distance
4.maximum distance

linkage:
1. single-linkage: min distance between any two points
2. complete linkage: max distance between any two points
3. ward: seeks to minimize the variance between clusters

 """

import numpy as np
import matplotlib.pyplot as plt 

from scipy.cluster.hierarchy import dendrogram, linkage



def main():
    D = 2
    s = 4
    mu1 = np.array([ 0,0])
    mu2 = np.array([ s,s])
    mu3 = np.array([0,s])

    N = 900
    X = np.zeros((N,D))
    X[:300,:] = np.random.randn(300,D) + mu1
    X[300:600,:] = np.random.randn(300,D) + mu2
    X[600:,:] = np.random.randn(300,D) + mu3

    Z =linkage(X, 'ward')
    print('Z.shape', Z.shape)

    plt.title("ward")
    dendrogram(Z)
    plt.show()

    Z =linkage(X, 'single')
    plt.title("single")
    dendrogram(Z)
    plt.show()

    Z =linkage(X, 'complete')
    plt.title("complete")
    dendrogram(Z)
    plt.show()


if __name__=="__main__": 
    main()
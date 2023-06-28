""" Kmeans Clustering is used to divide data into non-overlapping sub-group.it does well when the clusters
have a kind of sherical shape. it assumes spherical shapes of cluster and does not work well when clusters 
are in different shapes. it does not learn the number of clusters from the data and requires it to be
pre-defined.

HARD K_MEANS
TRAINING ALGORITHM
1. Initialize m(1), m(2), ......, m(k) = random points in X.pick random points in X to be our means
2. While not converge
3. Decide which cluster each point in X belong to
4. Recalculate cluster centres means based on the X's that were assigned to it(mean)
5. Repeat 3 and 4 until the result is found. i.e until the alogorithm converges. 
this happens in about 5 to 15 steps
This is different from gradient descent where we may need like thousand iterations

HARD K_MEANS AND SOFT K_MEANS
In hard K_means, a point can only belong to center but for soft k_means we measure the probability
 of a point belonging to two  or more points

SOFT K_MEANS
TRAINING ALGORITHM
1. Initialize m(1), m(2), ......, m(k) = random points in X.pick random points in X to be our means
2. While not converge
3. calculate cluster responsibilities:
4. Recalculate cluster centres means based on the X's that were assigned to it(mean) using cluster responsibility

"""

import numpy as np
import matplotlib.pyplot as plt

def d(u,u):
    diff = u-v
    return diff.dot(diff)

def cost(X,R,M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost += R[n,k]*d(M[k], X[n])
    return cost

        
def plot_k_means(X,k, max_iter =20, beta =1.0):
    N, D  = X.shape
    M = np.zeros((K,D))
    R = np.zeros((N,K))

    for k in range(k):
        M[K] = X[np.random.choice(N)]

    costs = np.zeros(max_iter)
    for i in range (max_iter):
        for k in range (k):
            for n in range (N):
                R[n,k] = np.exp(-beta*d(M[k], X[n]))/ np.sum(np.exp(-beta*d(M[j], X[n])) for j in range (k))
        for k in range (k):
            M[k] = R[:, k].dot(X)/ R[:, k].sum()

        costs[i] =cost(X, R,M)
        if i > 0:
            if np.abs(costs[i] - costs[i -1]) < 0.1:
                break

    plt.plot(costs)
    plt.title('costs')
    plt.show()

    random_colors = np.random.random((k,3))
    colors = R.dot(random_colors)
    plt.scatter (X[:, 0], X[:, 1], c = colors)
    plt.show()

        
def main():
    D = 2
    s = 4
    mu1 = np.array([ 0,0])
    mu2 = np.array([ s,s])
    mu3 = np.array([0,s])

    N = 900
    X =np.zeros((N,D))
    X[:300,:] = np.random.random(300,D) + mu1
    X[300:600,:] = np.random.random(300,D) + mu2
    X[600:,:] = np.random.random(300,D) + mu3

    plt.scatter(X[:,0],X[:,1])
    plt.show()

    k =3 
    plot_k_means(X,k)

    k =5 
    plot_k_means(X,k, max_iter = 0.3)

    k =5 
    plot_k_means(X,k, max_iter = 30, beta = 0.3)
    
if __name__=="__main__": 
    main()
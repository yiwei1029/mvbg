import math
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def dist_2m_sq(A,B):
    #assume A=(n,d_),B=(m,d_),expand A=>(n,m,d_),B=>(n,m,d_)
    n = A.shape[0]
    m = B.shape[0]
    # expand
    A_tile = np.tile(np.expand_dims(A,1),[1,m,1])
    B_tile = np.tile(np.expand_dims(B,0),[n,1,1])
    dist = np.sum(np.square(A_tile-B_tile),axis=-1)
    return dist

import scipy.io as sio
from sklearn.cluster import KMeans
def read_data(filename):
    data = sio.loadmat(filename)
    X = data['X']
    labels = data['Y']
    return X,labels

def cal_rbf_dist(A,B,n_neighbors,t):
    dist = dist_2m_sq(A,B)
    N =dist.shape[0]
    W = np.zeros((N,N))
    for i in range(N):
        idx = np.argsort(dist[i])[1:1+n_neighbors]
        W[i,idx] = np.exp(-1/t*dist[i][idx])
        # W[idx,i] = np.exp(-1/t*dist[idx][i])
        W[idx,i]= W[i,idx]
    return W

def eig_selection(cov,d_ ,top = True):
    '''
    top: top k eigenvalue
    '''
    eigval, eigvec = np.linalg.eig(cov)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    if top:
        idx = np.argpartition(eigval, -d_)[-d_:]
    else:
        idx = np.argpartition(eigval, d_)[:d_]
    selected_vec = eigvec[:,idx]
    return selected_vec

from sklearn.cluster import KMeans
def kmeans(Y,n_clusters):
    '''
    Y: dimension embedding
    '''
    km = KMeans(random_state=None,n_clusters=n_clusters).fit(Y.T)
    return km.labels_

def kmeans_cluster_prob(Y:np.ndarray,n_clusters:int =7)-> np.ndarray:
    '''
    Y:d*n (original data)
    '''
    km = KMeans(random_state=None,n_clusters=n_clusters).fit(Y.T)
    source   = Y.T
    centers = km.cluster_centers_
    dist = dist_2m_sq(source,centers)
    prob = dist/dist.sum(1).reshape(-1,1)
    return prob

def random_index(n,train_ratio):
    '''
    n: sample size
    '''
    train_size = math.floor(n*train_ratio)
    sample_idx = np.array(range(n))
    train_idx = np.random.choice(sample_idx,size=train_size,replace=False)
    test_idx = np.array(list(set(sample_idx) - set(train_idx)))
    return train_idx,test_idx

def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
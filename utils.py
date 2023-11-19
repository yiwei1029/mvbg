import numpy as np
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
    km = KMeans(random_state=None,n_clusters=n_clusters).fit(Y.T)
    source   = Y.T
    centers = km.cluster_centers_
    dist = dist_2m_sq(source,centers)
    prob = dist/dist.sum(1).reshape(-1,1)
    return prob

import numpy as np
from numpy.linalg import multi_dot
from sklearn import manifold
from utils import *
class MVBG:
    def __init__(self,alpha,gamma,beta):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def mvbg(self,w, X, Z,d_): #d_:embedding d; X is a list of (d_v,n)
        #step1: updating U_v
                #(v,d_v,n)
        v = len(w)
        U=[multi_dot([x,Z.T, np.linalg.inv(Z.dot(Z.T))]) for x in X]
        #step2: Updating E
        D_F=np.diag(np.sum(axis=0))
        D_G=np.diag(np.sum(axis=1))
        A_temp = np.linalg.inv(np.sqrt(D_F)).dot(Z).dot(np.linalg.inv(np.sqrt(D_G))) #for SVD
        u_A,s_A,v_A =np.linalg.svd(A_temp)
        F = u_A[:,:d_]*np.sqrt(2)/2
        G = v_A[:d_,:].T*np.sqrt(2)/2
        E = np.concatenate([F,G])
        #step3: Updating Z
        self.update_z(w,U, X, F,G,Z)
        #step4: Updating W
        h = [X[i]-U[i].dot(Z) for i in range(len(w))]
        h = [sum(h_v**2) for h_v in h]
        power = 1/(1-self.gamma)
        w = h**power/sum(h**power)


    def update_z(self,w,U, X, F,G,Z):
        F_ii = np.diagonal(dist_2m_sq(F,F))
        G_ii = np.diagonal(dist_2m_sq(G,G))
        #
        F_norm = F*1/F_ii.reshape(-1,1)
        G_norm = G*1/G_ii.reshape(-1,1)
        d_FG = dist_2m_sq(F_norm,G_norm)
        # U=(v,d_v,m)
        U_temp = sum(w**self.gamma*[u.T.dot(u) for u in U]) +self.alpha*np.identity(U[0].T.shape[0])
        V = 2*sum(w**self.gamma*[X[i].T.dot(U[i]) for i in range(len(w))])-self.beta*d_FG
        # main step: Z=(m,n)
        #init mu and rho
        mu=np.random.rand()
        rho = np.random.rand()+1
        for i in range(Z.shape[1]):

            # Fixing z_i and solving φ
            phi = Z[:,i]+1/mu*(eta-U_temp.T.dot(Z[:,i]))
            #Fixing φ and solving zi
            k = 1/mu*(mu*phi-eta-U_temp.dot(phi)+V[i,:].T)
            #find lmda
            for lmda in np.linspace(-np.abs(max(k)),np.abs(max(k)),10000):
                z_i = np.where(k+lmda>=0,k+lmda,0)
                if(np.abs(z_i.sum())<1e-3):
                    break
            
            eta = eta+mu*(z_i-phi)
            mu = rho*mu

            Z[:,i]=z_i #

class PCA:
    def cpca(self,X,d_): #X: list of (d_v,n)
        X = np.concatenate(X)
        X = X-X.mean(axis=1).reshape(-1,1)
        cov_X = X.dot(X.T)
        eigenval,eigenvec = np.linalg.eig(cov_X)
        eigenval = np.real(eigenval)
        eigenvec = np.real(eigenvec)    
        idx = np.argpartition(eigenval,-d_)[-d_:]
        select_vec = eigenvec[:,idx]
        out = select_vec.T.dot(X)
        return out

    def dpca(self,X,d_):
        dim_reductions =[]
        for x in X:
            x = x-x.mean(axis=1).reshape(-1,1)
            cov_x = x.dot(x.T)
            eigenval,eigenvec = np.linalg.eig(cov_x)
            eigenval = np.real(eigenval)
            eigenvec = np.real(eigenvec)    
            idx = np.argpartition(eigenval,-d_)[-d_:]
            select_vec = eigenvec[:,idx]
            out = select_vec.T.dot(x)
            dim_reductions.append(out)
        return sum(dim_reductions)/len(dim_reductions)

class LPP_LE:
    def lpp(self,X,t,d_):
        '''
        Locality Preserving Projection \n
        X: list of (d_v,n)
        '''
        X = np.concatenate(X)
        W = np.exp(1/t*dist_2m_sq(X.T,X.T))
        D = np.diag(W.sum(1))
        L  = D-W
        lhs = X.dot(L).dot(X.T)
        rhs = X.dot(D).dot(X.T)
        cov = np.linalg.inv(rhs).dot(lhs)
        eigval, eigvec = np.linalg.eig(cov)
        idx = np.argpartition(eigval, d_)[:d_]
        selected_vec = eigvec[:,idx]
        return selected_vec.T.dot(X)
    def le(self,X,d_,n_neighbors):
        X = np.concatenate(X)
        se = manifold.SpectralEmbedding(n_components=d_,n_neighbors=n_neighbors)
        Y = se.fit_transform(X.T)
        return Y.T
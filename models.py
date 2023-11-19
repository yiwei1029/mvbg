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
        W = cal_rbf_dist(X.T,X.T,10,t) #X here should be (n,d)
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

class MSE:
    def mse(self,X,gamma,d_,t,epoch =500):
        '''
        MSE ALgorithm \n
        X: list of (d_v,n)
        d_: embedding dimension
        gamma: regularization parameter 
        '''
        W = []
        n_v = len(X)
        for x in X:
            W.append(cal_rbf_dist(x.T,x.T,70,t=t))
        D =  [np.diag(w.sum(1)) for w in W]
        L =  [np.sqrt(np.linalg.inv(D[i])).dot(D[i]-W[i])\
              .dot(np.sqrt(np.linalg.inv(D[i]))) for i in range(n_v)]
        #init
        alpha = [1/n_v]*n_v
        # loop
        for i in range(epoch):
            # update Y
            L_weighted_sum  = sum([alpha[i]**gamma*L[i] for i in range(n_v)])
            Y = eig_selection(L_weighted_sum,d_=d_,top=False).T

            #update alpha
            alpha =  [1/np.trace(Y.dot(L[i]).dot(Y.T)) for i in range(n_v)]
            alpha = alpha/sum(alpha)
        return Y
    
class MDcR:
    def mdcr(self,X,d_,t,lmd,epoch =500):
        n_v = len(X)
        W =  []
        P=[]
        # init P_v
        for x in X:
            W.append(cal_rbf_dist(x.T,x.T,70,t=t))
        for v in range(n_v):
            x = X[v]
            w = W[v]
            d = np.diag(w.sum(1))

            l = np.sqrt(np.linalg.inv(d)).dot(w).dot(np.sqrt(np.linalg.inv(d)))
            cov =  x.dot(l).dot(x.T)
            selected_vec = eig_selection(cov,d_,True)
            P.append(selected_vec)
        
        #loop
        
        for i in range(epoch):
            Z = [ P[v].T.dot(X[v])for v in range(n_v) ]
            K = [z.T.dot(z) for z in Z]
            for v in range(n_v):
                x = X[v]
                l = np.sqrt(np.linalg.inv(d)).dot(w).dot(np.sqrt(np.linalg.inv(d)))

                A = x.dot(l).dot(x.T)
                h  = np.identity(x.shape[1])-1/x.shape[1]
                
                B = sum([x.dot(h).dot(K[j]).dot(h).dot(x.T) \
                         for j in range(n_v) if j!=v ]   )
                cov = (A+lmd*B)
                selected_vec = eig_selection(cov,d_,True)
                P[v] = selected_vec
        
        return sum([ P[v].T.dot(X[v])for v in range(n_v) ])

class DSE:
    def dse(self,X,k,d_,t,epoch =500):
        A_list=[]
        n_v= len(X)
        for x in X:
            pattern = kmeans_cluster_prob(x.T,k)  # (n,k)  
            A_list.append(pattern)
        A  = np.concatenate(A_list,axis=1) # 橫向拼接 (n,k*n_v)
        #init B and P
        B = np.random.rand(x.shape[1],k) #(n,k)
        B = B/B.sum(1).reshape(-1,1)
        P =  np.random.rand(k,x.shape[1]*n_v) #(k,k*n_v)
        #loop
        for i in range(epoch):
            #update B
            block1 = A.divide(B.dot(P)).dot(P.T).sum(1).reshape(-1,1)+alpha*(1/B.sum(1)).reshape(-1,1)
            block2 = (P.sum(1)+alpha).reshape(-1,1)
            B =   B*block1/block2
            #update P
            block3 = (A/(B.dot(P))).T.dot(B).sum(0).reshape(1,-1)
            block4 = B.sum(0).reshape(1,-1) # (1,k)
            P = P*block3/block4
        return B.argmax(1)
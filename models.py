from colorama import init
import numpy as np
from numpy.linalg import multi_dot
from sklearn import manifold
from utils import *
from tqdm import tqdm
import numpy.linalg as LA

class MVBG:
    def __init__(self,alpha,gamma,beta):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.name = "MVBG"

    def mvbg(self, X, m,d_,t,epoch): #d_:embedding d; X is a list of (d_v,n)
        '''
        m: anchor number
        '''
        w = np.array([1/len(X)]*len(X))
        n = X[0].shape[1]
        # Z = np.full(shape=(m,n),fill_value=1/m)
        Z = np.random.rand(m,n)
        # Z = Z/Z.sum(1).reshape(-1,1)
        for i in range(epoch):
            #step1: updating U_v=(d_v,m)
            v = len(w)
            U=[multi_dot([x,Z.T, np.linalg.inv(Z.dot(Z.T))]) for x in X]
            #step2: Updating E
            D_F=np.diag(np.sum(Z,axis=0))
            D_G=np.diag(np.sum(Z,axis=1))
            A_temp = np.linalg.inv(np.sqrt(D_F)).dot(Z.T).dot(np.linalg.inv(np.sqrt(D_G))) #for SVD
            # print(A_temp)
            u_A,s_A,v_A =np.linalg.svd(A_temp)
            F = u_A[:,:d_]*np.sqrt(2)/2
            G = v_A[:d_,:].T*np.sqrt(2)/2
            E = np.concatenate([F,G])
            #step3: Updating Z
            self.update_z(w,U, X,D_F,D_G, F,G,Z,t)
            #step4: Updating W
            h = [np.linalg.norm(X[i]-U[i].dot(Z)) for i in range(len(w))]
            h = np.array(h)
            # h = np.array([sum(h_v**2) for h_v in h])
            power = 1/(1-self.gamma)
            w = h**power/sum(h**power)
        return F.T

    def predict(self,X, m,d_,n_clusters,t,epoch):
        P = self.mvbg(X, m,d_,t,epoch)
        prob = kmeans(P,n_clusters)
        return prob
    
    def update_z(self,w,U, X, D_F,D_G,F,G,Z,t):
        # Z=(m,n)
        # dF_ii = np.diagonal(dist_2m_sq(F,F))
        # dG_ii = np.diagonal(dist_2m_sq(G,G))
    
        #
        F_norm = F/np.diagonal(D_F).reshape(-1,1)
        G_norm = G/np.diagonal(D_G).reshape(-1,1)
        d_FG = cal_rbf_dist(F_norm,G_norm,n_neighbors=10,t=t)
        # U=(v,d_v,m)
        U_temp = sum([w[i]**self.gamma*U[i].T.dot(U[i]) for i in range(len(w))]) +self.alpha*np.identity(U[0].T.shape[0])
        V = 2*sum([w[i]**self.gamma*X[i].T.dot(U[i]) for i in range(len(w))])-self.beta*d_FG
        # main step: Z=(m,n)
        #init mu and rho
        mu=1000 #防止U_temp隨alpha值變化導致SVD失敗
        rho = np.random.rand()+1
        eta =  np.random.rand()
        for i in range(Z.shape[1]):
            # Fixing z_i and solving φ
            phi = Z[:,i]+1/mu*(eta-U_temp.T.dot(Z[:,i]))
            #Fixing φ and solving zi
            k = 1/mu*(mu*phi-eta-U_temp.dot(phi)+V[i,:].T)
            # # find lmda
            # for lmda in np.linspace(-np.abs(max(k))-1/len(k),np.abs(max(k))+1/len(k),len(k)*1000):
            #     z_i = np.where(k+lmda>=0,k+lmda,0)
            #     if(np.abs(z_i.sum()-1)<1/1000):
            #         break
            z_i = np.exp(k/np.max(k))/sum(np.exp(k/np.max(k)))

            eta = eta+mu*(z_i-phi)
            mu = rho*mu
            Z[:,i]=z_i 
            # Z = Z/Z.sum(1).reshape(-1,1)
            # print(i,Z)

class DPCA:
    def __init__(self) -> None:
        self.name =   'DPCA'
    def dpca(self,X,d_):
        dim_reductions =[]
        P  = []
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
            P.append(select_vec)
        return P
    def predict(self,X_train,X_test,d_,k):
        P =self.dpca(X_train,d_)
        temp = [ P[i].T.dot(X_test[i]) for i in range(len(X_test))]
        dim_emb =  sum(temp)/len(temp)
        pred = kmeans(dim_emb,k)
        return pred

    
class CPCA:
    def __init__(self) -> None:
        self.name =  'CPCA'
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
        return select_vec
  
    def predict(self,X_train,X_test,d_,k):
        '''
        k: n clusters
        '''
        P = self.cpca(X_train,d_)
        dim_emb  = P.T.dot(np.concatenate(X_test))
        pred = kmeans(dim_emb,k)
        return pred


class LPP:
    def __init__(self) -> None:
        self.name =   'LPP'
    def lpp(self,X,t,d_,nn):
        '''
        Locality Preserving Projection \n
        X: list of (d_v,n)
        nn: n_neigbors
        '''
        
        # X = np.concatenate(X)
        P  = []
        for x in X:
            W = cal_rbf_dist(x.T,x.T,nn,t) #X here should be (n,d)
            D = np.diag(W.sum(1))
            L  = D-W
            lhs = x.dot(L).dot(x.T)
            rhs = x.dot(D).dot(x.T)
            cov = np.linalg.pinv(rhs).dot(lhs)
            eigval, eigvec = np.linalg.eig(cov)
            eigval = np.real(eigval)
            eigvec = np.real(eigvec)
            idx = np.argpartition(eigval, d_)[:d_]
            selected_vec = eigvec[:,idx]
            P.append(selected_vec)
        return P
    
    def predict(self,X_train,X_test,t,d_,k,nn=100):
        P = self.lpp(X_train,t,d_,nn)
        dim_emb = sum([P[i].T.dot(X_test[i]) for i in range(len(X_test))])
        pred = kmeans(dim_emb,k)
        return pred
    

class LE:
    def __init__(self) -> None:
        self.name =    'LE'
    def le(self,X,d_,n_neighbors):
        # X = np.concatenate(X)
        se = manifold.SpectralEmbedding(n_components=d_,n_neighbors=n_neighbors)
        # Y = se.fit_transform(X.T) # direct ouput dim_emb
        Y = [se.fit_transform(x.T) for x in X]
        return sum([y.T for y in Y]) #(d_,n)
    
    def predict(self,X, d_,n_neighbors,k):
        dim_emb =  self.le(X,d_,n_neighbors)
        pred = kmeans(dim_emb,k)
        return pred

class MSE:
    def __init__(self) -> None:
        self.name='MSE'
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
    def predict(self,X,gamma,d_,t,k,epoch =500):
        dim_emb   = self.mse(X,gamma,d_,t,epoch)
        pred =  kmeans(dim_emb,k)
        return pred


class MDcR:
    def __init__(self) -> None:
        
        self.name   = 'MDcR'
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
        return P
    
    def predict(self,X,d_,t,lmd,nn,epoch=500):
        P = self.mdcr(X,d_,t,lmd,epoch)
        n_v = len(X)
        dim_emb = sum([ P[v].T.dot(X[v])for v in range(n_v) ])
        pred = kmeans(dim_emb,nn)
        return pred
    

class DSE:
    def __init__(self) -> None:
        self.name  = 'DSE'
    def dse(self,X,k,alpha,epoch =500):
        A_list=[]
        n_v= len(X)
        for x in X:
            pattern = kmeans_cluster_prob(x,k)  # (n,k)  
            A_list.append(pattern)
        A  = np.concatenate(A_list,axis=1) # 橫向拼接 (n,k*n_v)
        #init B and P
        B = np.random.rand(x.shape[1],k)*k #(n,k)
        # B =    np.ones_like(B)
        B = np.exp(B)/(np.exp(B).sum(1).reshape(-1,1))
        P =  np.random.rand(k,k*n_v) #(k,k*n_v)
        # P =   np.ones_like(P)
        #loop
        for i in range(epoch):
            #update B
            block1 = (A/(B.dot(P))).dot(P.T)+alpha*(1/B.sum(1)).reshape(-1,1) #(n,k)
            block2 = (P.sum(1)+alpha).reshape(-1,1) #(k,1)
            B =   B*block1/block2.reshape(1,-1)
            #update P
            block3 =  B.T.dot(A/(B.dot(P)))
            block4 = B.sum(0).reshape(-1,1) # (k,1 )
            P = P*block3/block4
            b = B.argmax(1)
        return b
    def predict(self,X,k,alpha,epoch =500):
        return self.dse(X,k,alpha,epoch)


class MVP:
    def __init__(self) -> None:
        self.name=   "MVP"
    def mvp(self,X,r,alpha,t,d_,epoch=500):
        '''
        t for rbf
        x: (n,d)
        '''
        n_v = len(X)
        X  = [x.T for x in X]
        w = np.array([1/n_v]*n_v)
        W =  []
        for x in X:
            W.append(cal_rbf_dist(x,x,70,t=t))
        D = [np.diag(w.sum(1))  for w in W]
        L =  [np.identity(D[i].shape[0])-np.linalg.inv(np.sqrt(D[i])).dot(W[i])\
              .dot(np.linalg.inv(np.sqrt(D[i]))) for i in range(n_v)]
        L_sum = sum([alpha*w[i]*L[i]  for  i in range(n_v)])
        
        A  = [x.dot(np.linalg.pinv(x.T.dot(x))).dot(x.T) for x in X]
        # A  = [x.dot(x.T) for x in X]

        A_coef = [np.abs(np.diag(a.sum(1))/np.max(a)) for a in A]
        delta_star = [LA.inv(np.sqrt(A_coef[i])).dot(A[i])\
                      .dot(LA.inv(np.sqrt(A_coef[i]))) for i in range(n_v)]
        delta = [np.identity(delta_star[0].shape[0])-d_s for d_s in delta_star]
        delta_sum = sum(delta)
        
        for i in range(epoch):
            #update U
            cov =  delta_sum+L_sum
            U = eig_selection(cov,d_,False)
            #update w
            up = [(1/np.trace(U.T.dot(L[i]).dot(U) ) )**(1/r-1)   for i in range(n_v)]
            down =  sum(up)
            w = up/down
            #update L_sum
            L =  [np.identity(D[i].shape[0])-np.linalg.inv(np.sqrt(D[i])).dot(W[i])\
              .dot(np.linalg.inv(np.sqrt(D[i]))) for i in range(n_v)]
            L_sum = sum([alpha*w[i]*L[i]  for  i in range(n_v)])
        return U
    
    def predict(self,X,r,alpha,t,d_,k,epoch=500):
        dim_emb = self.mvp(X,r,alpha,t,d_,epoch).T
        pred = kmeans(dim_emb,k)
        return pred

class BASE:
    def __init__(self) -> None:
        self.name  =   'BASE'
    def base(self,X,n_clusters):
        X = np.concatenate(X)
        return kmeans(X,n_clusters)
    def predict(self,X,n_clusters):
        return self.base(X,n_clusters)
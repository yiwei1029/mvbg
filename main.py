from os import name
import warnings
warnings.filterwarnings('ignore')
#model and 
import math
from utils import *
from tqdm import tqdm
from models import MSE,MDcR,DSE,MVBG,CPCA,DPCA,MVP,LPP,LE
import scipy
def main_cal(model_name,params):
    datasets_names = ['BBC','MSRC-v1','NGs','Reuters','YALE']
    for dt_name in datasets_names:
        # data
        X,labels=read_data(f'datasets/data sets/{dt_name}.mat')
        X = [X[0,i].T for  i  in range(X.shape[1])]
        X =  [x[:300,] for x in X] #d<=300 ,n = X[0].shape[1]
        if isinstance(X[0],scipy.sparse._csr.csr_matrix):
            X =   [np.asarray(x.todense()) for x in X]
        labels  = labels.squeeze()
        k = len(set(labels))
        #models
        #main loop
        res_nmi = [];res_acc =[];res_ari=[];res_purity  = []
        train_ratio=0.1
        d_max  =  min(math.floor((1-train_ratio)*X[0].shape[1]),int(np.min([x.shape[0] for x in X])) ) #min(d,測試x的維數)
        d_range  = range(k,d_max)
        model  =  model_name

        for d_ in tqdm(d_range): 
            nmi_list = [];acc_list =[]; ari_list=[]; purity_list=[]
            for i in range(5):
                # train_test split
                train_idx,test_idx = random_index(X[0].shape[1],train_ratio)
                X_train = [x[:,train_idx] for x in X]
                X_test = [x[:,test_idx] for x in X]
                y_test = labels[test_idx]
                #train

                # pred =  model.predict(X_test,0.5,2,1e6,d_,k,10) #MVP ok
                # pred   = model.predict(X_train,X_test,1e7,d_,k,200) #lPP ok 
                # pred  = model.predict(X_test,d_,20,k) #LE ok
                # pred = model.predict(X_train,X_test,d_,k) #DPCA ok
                # pred =  model.predict(X_test,0.5,d_,1e8,k,100) #MSE ok
                # pred   =  model.predict(X_test,d_,1e8,5,100,100)#mdcr ok
                # pred = model.predict(X_test,d_max,d_,k,1e8,30)# m: >=dmax ok mvbg
                # pred = model.dse(X_test,k,2,100) #dse 
                # pred  = model.predict(X_train, X_test,d_,k) #CPCA
                pred = eval(model_name+'.predict'+params)
                # criterion
                #nmi
                from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_rand_score

                nmi = normalized_mutual_info_score(y_test,pred)
                nmi_list.append(nmi)
                #acc
                acc =   acc_score(y_test,pred)
                acc_list.append(acc)
                #ari
                ari =   adjusted_rand_score(y_test,pred)
                ari_list.append(ari)
                #purity
                purity  =  purity_score(y_test,pred)
                purity_list.append(purity)
            #
            nmi = np.mean(nmi_list)
            acc  = np.mean(acc_list)
            ari = np.mean(ari_list)
            purity = np.mean(purity_list)
            # 
            res_nmi.append(nmi)
            res_acc.append(acc)
            res_ari.append(ari)
            res_purity.append(purity)
        np.save(f'./result/nmi/{dt_name}/{model.name}.npy',res_nmi)
        np.save(f'./result/acc/{dt_name}/{model.name}.npy',res_acc)
        np.save(f'./result/ari/{dt_name}/{model.name}.npy',res_ari)
        np.save(f'./result/purity/{dt_name}/{model.name}.npy',res_purity)

if __name__=='__main__':
    model_params_dict={'MSE()':'(X_test,0.5,d_,1e8,k,100)',
                        'MDcR()':'(X_test,d_,1e8,5,100,100)',
                        'DSE()':'(X_test,k,2,100)',
                        'MVBG(0.1,2,0.1)':'(X_test,d_max,d_,k,1e8,30)',
                        'CPCA()':'(X_train, X_test,d_,k)',
                        'DPCA()':'(X_train,X_test,d_,k)',
                        'MVP()':'(X_test,0.5,2,1e6,d_,k,10)',
                        'LPP()':'(X_train,X_test,1e7,d_,k,200)',
                        'LE()':'(X_test,d_,20,k)'}
    for model,params in model_params_dict.items():
        main_cal(model,params)

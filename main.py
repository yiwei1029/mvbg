from os import name
import warnings
warnings.filterwarnings('ignore')
import math
from utils import *
from tqdm import tqdm
from models import MSE,MDcR,DSE,MVBG,CPCA,DPCA,MVP,LPP,LE,BASE
import scipy
from multiprocessing import Pool
import multiprocessing
import os
def main_cal(model_name,params,dt_name):
    # print('Run task %s (%s) (%s)...' % (model_name, dt_name,os.getpid()))
    
    # data
    X,labels=read_data(f'datasets/mats/{dt_name}.mat')
    X = [X[0,i].T for  i  in range(X.shape[1])]
    X =  [x[:300,:500] for x in X] #d<=300 ,n = X[0].shape[1]
    if isinstance(X[0],scipy.sparse._csr.csr_matrix):
        X =   [np.asarray(x.todense()) for x in X]
    labels  = labels.squeeze()
    k = len(set(labels))
    #models
    #main loop
    res_nmi = [];res_acc =[];res_ari=[];res_purity  = []
    train_ratio=0.5
    d_max  =  min(math.floor((1-train_ratio)*X[0].shape[1]),\
                  int(min([x.shape[0] for x in X])) ) #min(n_test,d_min_X)
    d_range  = range(k,min(d_max,k+30),2)
    model  =  model_name
    

    for d_ in tqdm(d_range):
        nmi_list = [];acc_list =[]; ari_list=[]; purity_list=[]
        for i in range(5):
            # train_test split
            train_idx,test_idx = random_index(X[0].shape[1],train_ratio)
            X_train = [x[:,train_idx] for x in X]
            X_test = [x[:,test_idx] for x in X]
            y_test = labels[test_idx]
            #train and predict
            pred = eval(model_name+'.predict'+params)
            # criterion
            from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_rand_score
            #nmi
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
        nmi_ = np.mean(nmi_list)
        acc_  = np.mean(acc_list)
        ari_ = np.mean(ari_list)
        purity_ = np.mean(purity_list)
        # 
        res_nmi.append(nmi_)
        res_acc.append(acc_)
        res_ari.append(ari_)
        res_purity.append(purity_)
        # print('./result/nmi/{}/{}.npy'.format(dt_name, eval(model+'.name')))
    for indicator in ['nmi','acc','ari','purity']:
        if not (os.path.exists('./result/{}/{}/'.format(indicator,dt_name))):
            os.makedirs('./result/{}/{}/'.format(indicator,dt_name))
        np.save('./result/{}/{}/{}.npy'.format(indicator,dt_name, \
                                               eval(model+'.name')),eval('res_'+indicator))
        np.save('./result/{}/{}/{}.npy'.format(indicator,dt_name,'d_range'),np.array(d_range))
        
def err_call_back(err):
        print(f'error：{str(err)} ')
import traceback

def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)
class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            # Here we add some debugging help. If multiprocessing's debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can clean up
            raise
            # It was fine, give a normal answer
        return result
    pass
if __name__=='__main__':
    model_params_dict={'BASE()':'(X_test,k)',
                        'MSE()':'(X_test,0.5,d_,1e7,k,10)',
                        'MDcR()':'(X_test,d_,1e7,5,k,10)',
                        'DSE()':'(X_test,k,2,10)',
                        'MVBG(0.1,2,0.1)':'(X_test,60,d_,k,1e7,10)',
                        'CPCA()':'(X_train, X_test,d_,k)',
                        'DPCA()':'(X_train,X_test,d_,k)',
                        'MVP()':'(X_test,0.5,2,1e7,d_,k,10)',
                        'LPP()':'(X_train,X_test,1e7,d_,k,20)',
                        'LE()':'(X_test,d_,20,k)'
                        }
    # model_params_dict= {'MVBG(0.1,2,0.1)':'(X_test,60,d_,k,1e7,10)'}
    datasets_names = ['BBC','MSRC-v1','NGs','Reuters','YALE']
    # datasets_names = ['BBC'] #

    multiprocessing.log_to_stderr()  # 加上此行
    p = Pool(2)
    
    for model,params in model_params_dict.items():
        for dt_name in datasets_names:
            p.apply_async(LogExceptions(main_cal), args=(model,params,dt_name))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
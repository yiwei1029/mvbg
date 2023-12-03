# from utils import *
# import sklearn
# x,y = read_data('datasets/mats/MSRC-v1.mat')
# print(x[0,1].shape)

import numpy as np
import random
import time
random.seed(10)
a = np.random.poisson(2,500)
print(a)
x =np.mean(a)
res = np.where(a+x>0,a+x,0).sum()

time1 = time.time()
while(np.abs(res-1)>0.1):
    x-=abs(res-1)/len(a)
    res = np.where(a+x>0,a+x,0).sum()
    print(res)
print(time.time()-time1)

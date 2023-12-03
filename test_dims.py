# from utils import *
# import sklearn
# x,y = read_data('datasets/mats/MSRC-v1.mat')
# print(x[0,1].shape)

import numpy as np
import random
import time
random.seed(10)
a = np.random.randn(30)
x =np.mean(a)
res = np.where(a+x>0,a+x,0).sum()

time1 = time.time()
count = 0
while(np.abs(res-1)>0.01):
    x-=(res-1)/(a+x>0).sum()
    res = np.where(a+x>0,a+x,0).sum()
    count+=1
print('time:{}, count:{}'.format((time.time()-time1),count))

from utils import *
res = find_sum_relu_x(a,x)
print(f'sum  is {res.sum()}')

# %%
import numpy as np
def cpca(X,d_): #X: list of (d_v,n)
    X = np.concatenate(X)
    X = X-X.mean(axis=1).reshape(-1,1)
    cov_X = X.dot(X.T)
    eigenval,eigenvec = np.linalg.eig(cov_X)    
    select_vec = eigenvec[:,np.partition(eigenval,-d_)[-d_:]]
    out = select_vec.T.dot(X)
    return out







# %%
import tarfile
import os
from tqdm import tqdm
import numpy as np
def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False

# untar('datasets/msrcorid.tar.gz','./datasets/')
from PIL import Image
def read_imgs():
    path = 'datasets/msrcorid'
    filepaths = []
    img_arrays = []
    labels = []
    for root, dirs, files in tqdm(os.walk(path)):
        # print(dirnames)     
        for file in files:
            if(file.endswith('JPG')):                         # 遍历文件
                file_path = os.path.join(root, file)
                img = np.asarray(Image.open(file_path))
                img_arrays.append(img.transpose((2,0,1)).reshape((3,-1,1)))
                labels.append(file[:3])   # 获取文件绝对路径  
                filepaths.append(file_path)            # 将文件路径添加进列表
        # for dir in dirs:                           # 遍历目录下的子目录
        #     dir_path = os.path.join(root, dir)     # 获取子目录路径
        #     all_files_path(dir_path)               # 递归调用    
    return img_arrays,labels

img_arrays,labels = read_imgs()
img_tensor = np.concatenate(img_arrays[:200],axis =2)
img_tensor.shape 



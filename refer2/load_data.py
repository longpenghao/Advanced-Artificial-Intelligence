import os
import gzip
import numpy as np


def load_data(data_folder):

    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
   
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
       
    # 读取每个文件夹的数据    
    with gzip.open(paths[0], 'rb') as lbpath:
        train_y_orig = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        train_len = train_y_orig.shape[0]
        train_y = np.zeros((10, train_len))
        for i in range(train_len):
            train_y[train_y_orig[i], i] = 1
        
     
    with gzip.open(paths[1], 'rb') as imgpath:
        train_x_orig = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(train_len, 784)
        train_x = train_x_orig.T
        train_x = train_x / 255
       
    with gzip.open(paths[2], 'rb') as lbpath:
        test_y_orig = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        test_len = test_y_orig.shape[0]
        test_y = np.zeros((10, test_len))
        for i in range(test_len):
            test_y[test_y_orig[i], i] = 1
       
    with gzip.open(paths[3], 'rb') as imgpath:
        test_x_orig = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(test_len, 784)
        test_x = test_x_orig.T
        test_x = test_x / 255
       
    return train_x, train_y, test_x, test_y, train_y_orig, test_y_orig
import numpy as np
from load_data import load_data
from sklearn.neighbors import KNeighborsClassifier
import time

if __name__ == "__main__":
    train_x, train_y, test_x, test_y, train_y_orig, test_y_orig = load_data("/home/pyj/homework/aai/dataset/MNIST/raw")
    train_x_sk = train_x.T
    test_x_sk = test_x.T
    train_y_sk = train_y_orig.T # 0-9
    test_y_sk = test_y_orig # 0-9

    for k in [1, 3, 5, 7]:
        knn_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=k)  
        
        knn_clf.fit(train_x_sk, train_y_sk)
        t1 = time.time()
        predict = knn_clf.predict(test_x_sk)
        t2 = time.time()
        correct_num = np.sum(predict == test_y_sk)
        test_sum = len(test_y_sk)
        incorrect_num = test_sum - correct_num
        print(f"K:{k}, incorrect_num:{incorrect_num}, accuracy:{round(correct_num / float(test_sum), 4)}, time:{round((t2 - t1), 4)}s")
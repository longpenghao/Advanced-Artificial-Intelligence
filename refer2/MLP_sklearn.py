import numpy as np
from load_data import load_data
from sklearn.neural_network import MLPClassifier
import time

if __name__ == "__main__":
    train_x, train_y, test_x, test_y, train_y_orig, test_y_orig = load_data("/home/pyj/homework/aai/dataset/MNIST/raw")
    train_x_sk = train_x.T
    test_x_sk = test_x.T
    train_y_sk = train_y_orig.T # 0-9
    test_y_sk = test_y_orig # 0-9
    # print(train_y_orig.shape, train_y_sk.shape)

    for n_h in [500, 1000, 1500, 2000]:
        mlp = MLPClassifier(hidden_layer_sizes=(n_h,), activation='relu', solver='adam',
                            learning_rate_init=0.001, max_iter=100)
        t1 = time.time()
        mlp.fit(train_x_sk, train_y_sk) 
        t2 = time.time()
        predict = mlp.predict(test_x_sk)
        correct_num = np.sum(predict == test_y_sk)
        test_sum = len(test_y_sk)
        incorrect_num = test_sum - correct_num
        accuracy = round(correct_num / float(test_sum), 4)
        print(f"hidden_layer_node:{n_h}, learning_rate:0.001, incorrect_num:{incorrect_num}, accuracy:{accuracy}, training_time:{round((t2 - t1), 4)}s")
    
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        mlp = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', solver='adam',
                            learning_rate_init=0.001, max_iter=100)
        t1 = time.time()
        mlp.fit(train_x_sk, train_y_sk) 
        t2 = time.time()
        predict = mlp.predict(test_x_sk)
        correct_num = np.sum(predict == test_y_sk)
        test_sum = len(test_y_sk)
        incorrect_num = test_sum - correct_num
        accuracy = round(correct_num / float(test_sum), 4)
        print(f"hidden_layer_node:1000, learning_rate:{lr}, incorrect_num:{incorrect_num}, accuracy:{accuracy}, training_time:{round((t2 - t1), 4)}s")

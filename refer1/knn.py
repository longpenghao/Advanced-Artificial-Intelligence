from sklearn.metrics import accuracy_score
from sklearn import naive_bayes

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import os

data_folder = 'data'

train_label = 'train-labels-idx1-ubyte'
train_image = 'train-images-idx3-ubyte'
test_label = 't10k-labels-idx1-ubyte'
test_image = 't10k-images-idx3-ubyte'

def load_data():
    path_y_train = os.path.join(data_folder, train_label)
    with open(path_y_train, 'rb') as raw_data:
        y_train = np.frombuffer(raw_data.read(), np.uint8, offset=8)

    path_x_train = os.path.join(data_folder, train_image)
    with open(path_x_train, 'rb') as raw_data:
        x_train = np.frombuffer(raw_data.read(), np.uint8, offset=16).reshape(len(y_train), 784)

    path_y_test = os.path.join(data_folder, test_label)
    with open(path_y_test, 'rb') as raw_data:
        y_test = np.frombuffer(raw_data.read(), np.uint8, offset=8)

    path_x_test = os.path.join(data_folder, test_image)
    with open(path_x_test, 'rb') as raw_data:
        x_test = np.frombuffer(raw_data.read(), np.uint8, offset=16).reshape(len(y_test), 784)


    return x_train, x_test, y_train, y_test

#knn算法
def knn():
    x_train, x_test, y_train, y_test = load_data()

    bayes = naive_bayes.GaussianNB()
    bayes.fit(x_train, y_train)
    bayes_score = accuracy_score(y_test, bayes.predict(x_test))
    print("Bayes accuracy:", bayes_score)

    knn = KNeighborsClassifier(n_neighbors = 7)
    knn.fit(x_train, y_train)
    knn_score = knn.score(x_test, y_test)
    print("KNN accuracy:", knn_score)


#全连接神经网络
def neural_network():
    x_train, x_test, y_train, y_test = load_data()
    mlp = MLPClassifier(solver='lbfgs',max_iter =500, alpha=1e-3,hidden_layer_sizes=(1000,),learning_rate_init=0.001)
    mlp.fit(x_train, y_train)

    #打分
    result = mlp.score(x_test, y_test)
    print(result)




if __name__ == "__main__":
    #___knn算法
    #knn()
    #____sklearn库实现的全连接神经网络
    neural_network()


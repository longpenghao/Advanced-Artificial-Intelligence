import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
    
def batch_data(X, Y, batch_size):
    total = X.shape[1]
    batch_num = total / batch_size
    batch_num = int(batch_num)
    if total % batch_size != 0:
        batch_num += 1
    X_Y_batch = []
    # print(batch_num)
    for i in range(batch_num):
        X_Y_batch.append((X[:, i*batch_size:(i+1)*batch_size], Y[:, i*batch_size : (i+1)*batch_size]))
    return X_Y_batch, batch_num


def sigmoid(Z):  
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):   
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)

    return A

def softmax(Z):
    """
    softmax激活函数
    手写数字识别输出层,一共有10个类别,使用softmax
    """
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def relu_backward(dA, cache):   
    Z = cache
    dZ = np.array(dA, copy=True) 

    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(A2, Y):
    """
    损失函数是10个输出神经元a的函数 经过计算直接求得L2中dZ用A、Y表示的值
    """
    dZ = A2 - Y
    return dZ

def initialize_parameters(n_x, n_h, n_y):   
    # np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def compute_cost(AL, Y):    
    m = Y.shape[1]
    cost = - (1./m) * (np.sum(Y * np.log(AL)))
    
    cost = np.squeeze(cost) 
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):    
    m = X.shape[1]
    
    # Forward propagation
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1, _ = linear_forward(X, W1, b1)
    A1 = relu(Z1)
    Z2, _ = linear_forward(A1, W2, b2)
    A2  = softmax(Z2)
    probas = A2
    p = np.argmax(probas, axis=0)
    incorrect_num = m - np.sum((p == y))
    accuracy = round(np.sum((p == y)/m), 4)
    return incorrect_num, accuracy

def two_layer_model(X, Y, layers_dims, learning_rate, batch_size = 1000, num_iterations = 1000, print_cost=True):
    # np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    X_Y_batch, batch_num = batch_data(X, Y, batch_size)
    # Loop (gradient descent)

    for i in range(0, num_iterations):
        total_batch_cost = 0
        for bat in range(batch_num):
            idX, idY = X_Y_batch[bat]

            # forward prop
            Z1, linear_cache1 = linear_forward(idX, W1, b1)
            A1 = relu(Z1)
            Z2, linear_cache2 = linear_forward(A1, W2, b2)
            A2  = softmax(Z2)

            # Compute cost
            cost = compute_cost(A2, idY)
            total_batch_cost += cost
            
            # backward prop
            dZ2 = softmax_backward(A2, idY)
            dA1, dW2, db2 = linear_backward(dZ2, linear_cache2)
            dZ1 = relu_backward(dA1, Z1)
            dA0, dW1, db1 = linear_backward(dZ1, linear_cache1)
            
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate = learning_rate)
            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            # print("one batch done")
        total_batch_cost /= batch_num
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(total_batch_cost)))
        if print_cost and i % 100 == 0:
            costs.append(total_batch_cost)
    return parameters

import time

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, train_y_orig, test_y_orig = load_data("/home/pyj/homework/aai/dataset/MNIST/raw")
    # m_train = train_x.shape[1]
    # m_test = test_x.shape[1]
    # print(m_train, m_test)
    n_x = 784
    n_y = 10

    # learning_rate = 0.005 by default
    for n_h in [500, 1000, 1500, 2000]:
        t1 = time.time()
        parameters = two_layer_model(train_x, train_y, learning_rate=0.005,layers_dims = (n_x, n_h, n_y), num_iterations = 500, print_cost=False)
        t2 = time.time()
        incorrect_num, pred = predict(test_x, test_y_orig, parameters)
        print(f"hidden_layer_node:{n_h}, learning_rate:0.001, incorrect_num:{incorrect_num}, accuracy:{pred}, training_time:{round((t2 - t1), 4)}s")

    # n_h = 1000 by default
    n_h = 1000
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        t1 = time.time()
        parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), learning_rate = lr, num_iterations = 500, print_cost=False)
        t2 = time.time()
        incorrect_num, pred = predict(test_x, test_y_orig, parameters)
        print(f"hidden_layer_node:1000, learning_rate:{lr}, incorrect_num:{incorrect_num}, accuracy:{pred}, training_time:{round((t2 - t1), 4)}s")

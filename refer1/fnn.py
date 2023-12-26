import os
import numpy as np

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


    y_one_hot = np.zeros((y_train.shape[0], 10))
    y_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_hot = np.zeros((y_test.shape[0], 10))
    y_hot[np.arange(y_test.shape[0]), y_test] = 1
    return x_train/2550, x_test/2550, y_one_hot, y_hot

class neuralNetwork:
    def __init__(self,numNeuronLayers,numNeurons_perLayer,learningrate):
        self.numNeurons_perLayer=numNeurons_perLayer
        self.numNeuronLayers=numNeuronLayers
        self.learningrate = learningrate
        self.weight=[]
        for i in range(numNeuronLayers):
            self.weight.append(np.random.normal(0.0, pow(self.numNeurons_perLayer[i+1],-0.5),  (self.numNeurons_perLayer[i+1],self.numNeurons_perLayer[i]) )  )
        self.activation_function = lambda x: 1.0/(1.0+np.exp(-x))

    def update(self,inputnodes,targets):     
        inputs = np.array(inputnodes,ndmin=2).T
        targets = np.array(targets,ndmin=2).T
        #前向传播
        #定义输出值列表（outputs[0]为输入值）
        self.outputs=[]
        self.outputs.append(inputs)
        #用激活函数对神经网络的每一层计算输出值，并保存到outputs列表中
        for i in range(self.numNeuronLayers):
            temp_inputs=np.dot(self.weight[i],inputs)
            temp_outputs=self.activation_function(temp_inputs)
            inputs=temp_outputs
            self.outputs.append(temp_outputs)
        #计算每层的训练误差
        self.output_errors=[]
        for i in range(self.numNeuronLayers):
            #输出层的误差=目标值-输出值
            if i == 0:
                self.output_errors.append(targets - self.outputs[-1])
            #隐藏层的误差=当前隐藏层与下一层之间的权值矩阵与下一层的误差矩阵的乘积
            else:
                self.output_errors.append(np.dot((self.weight[self.numNeuronLayers-i]).T, 
                                                    self.output_errors[i-1]))
        #反向传播
        for i in range(self.numNeuronLayers):
            #权值更新规则为之前权值+学习率*误差*第二层输出*（1-第二层输出）*第一层输出
            #f(x)*（1-f(x)）即为激活函数f(x)的导函数
            self.weight[self.numNeuronLayers-i-1] += self.learningrate * np.dot((self.output_errors[i] 
                * self.outputs[-1-i] * (1.0 - self.outputs[-1-i])), np.transpose(self.outputs[-1-i-1]))
    def test(self,test_inputnodes,test_labels):
        inputs = np.array(test_inputnodes,ndmin=2).T
        #走一遍前向传播得到输出
        for i in range(self.numNeuronLayers):
            temp_inputs=np.dot(self.weight[i],inputs)
            temp_outputs=self.activation_function(temp_inputs)
            inputs=temp_outputs
        #返回模型输出结果是否与测试用例标签一致
        return list(inputs).index(max(list(inputs)))==list(test_labels).index(1)

    def train(self, image_train, label_train, epoches):
        for i in range(epoches):
            print(i)
            for i in range(len(image_train)):
                self.update(image_train[i],label_train[i])

    def output(self,test_inputnodes,f):
        inputs = np.array(test_inputnodes,ndmin=2).T
        #走一遍前向传播得到输出
        for i in range(self.numNeuronLayers):
            temp_inputs=np.dot(self.weight[i],inputs)
            temp_outputs=self.activation_function(temp_inputs)
            inputs=temp_outputs
        
            f.write(str(list(inputs).index(max(list(inputs))))+' ')

if __name__ == '__main__': 
    learning_rate = 0.0001
    images_data, test_images_data, labels, test_labels = load_data()
    #1000时 acc=95.58
    ls=[784,500,10]
    n = neuralNetwork(2,ls,learning_rate)
    #轮数
    n.train(images_data, labels, epoches=100)
    # for j in range(10):
    #     for i in range(len(images_data)):
    #         n.update(images_data[i],labels[i])
    count = 0
    for i in range(len(test_images_data)):     
        # n.output(test_images_data[i],f)
        if n.test(test_images_data[i],test_labels[i]):
            count += 1
    print(count/10000)

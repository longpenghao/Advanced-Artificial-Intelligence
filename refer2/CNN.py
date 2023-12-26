import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import numpy as np
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.prediction = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output


if __name__ == "__main__":
    
    train_data = torchvision.datasets.MNIST(
        root='/home/pyj/homework/aai/dataset/', 
        train = True, 
        transform = torchvision.transforms.ToTensor(), 
        download = False
    )
    test_data = torchvision.datasets.MNIST(
        root='/home/pyj/homework/aai/dataset/', 
        train = False
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2 )
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255.
    test_y = test_data.test_labels
    
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), 0.001)
    loss_func = nn.CrossEntropyLoss()
    
    t1 = time.time()
    for epoch in range(30): # EPOCH = 30
        for i, (batch_x, batch_y) in enumerate(train_loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    t2 = time.time()

    test_output = cnn(test_x)
    pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
    incorrect_num = len(test_y) - np.sum(np.array(test_y) == pred)
    accuracy = round(np.sum(np.array(test_y) == pred) / len(test_y), 4)
    print(f"incorrect_num:{incorrect_num}, accuracy:{accuracy}, training_time:{round((t2 - t1), 4)}s")

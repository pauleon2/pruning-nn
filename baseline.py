import torch
import torch.nn as nn
import pandas as pd

from util import *


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)


class DropoutNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_ratio):
        super(DropoutNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.dropout(out)


def wd():
    train_set, valid_set = dataloader.get_train_valid_dataset()
    test_set = dataloader.get_test_dataset()

    s = pd.DataFrame(columns=['accuracy', 'weight decay', 'run'])

    for i in range(25):
        for wd in [0.05, 0.01, 0.005, 0.001]:
            model = NeuralNetwork(28 * 28, 100, 10)
            lr = 0.01
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
            loss_func = nn.CrossEntropyLoss()

            # train and test the network
            t = True
            epoch = 0
            p_acc = 0

            while t and epoch < 200:
                learning.train(train_set, model, optimizer, loss_func)
                new_acc = learning.test(valid_set, model)

                if new_acc - p_acc < 0.00001:
                    if lr > 0.0001:
                        # adjust learning rate
                        lr = lr * 0.1
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
                    else:
                        # stop training
                        t = False
                        break

                epoch += 1
                p_acc = new_acc

            acc = learning.test(test_set, model)
            tmp = pd.DataFrame({
                'accuracy': [acc],
                'weight decay': [wd],
                'run': [i]
            })
            s = s.append(tmp, ignore_index=True)
        s.to_pickle('./out/result/weight-decay.pkl')


def dropout():
    train_set, valid_set = dataloader.get_train_valid_dataset()
    test_set = dataloader.get_test_dataset()

    s = pd.DataFrame(columns=['accuracy', 'dropout', 'run'])

    for i in range(25):
        for drop_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            model = DropoutNeuralNetwork(28*28, 100, 10, drop_ratio)

            lr = 0.01
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
            loss_func = nn.CrossEntropyLoss()
            # train and test the network
            t = True
            epoch = 0
            p_acc = 0

            while t and epoch < 200:
                learning.train(train_set, model, optimizer, loss_func)
                new_acc = learning.test(valid_set, model)

                if new_acc - p_acc < 0.00001:
                    if lr > 0.0001:
                        # adjust learning rate
                        lr = lr * 0.1
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
                    else:
                        # stop training
                        t = False
                        break

                epoch += 1
                p_acc = new_acc

            acc = learning.test(test_set, model)
            tmp = pd.DataFrame({
                'accuracy': [acc],
                'dropout ratio': [wd],
                'run': [i]
            })
            s = s.append(tmp, ignore_index=True)
        s.to_pickle('./out/result/dropout.pkl')


if __name__ == '__main__':
    wd()
    dropout()

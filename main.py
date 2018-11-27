import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pruninig_nn.network import NeuralNetwork
from pruninig_nn.pruning import PruneNeuralNetStrategy, magnitude_based_pruning
from pruninig_nn.util import train, test

# constant variables
hyper_params = {
    'pruning_percentage': 90,  # percentage of weights pruned
    'pruning_update_rate': 2,
    'batch_size': 64,
    'test_batch_size': 100,
    'num_retrain_epochs': 10,
    'num_epochs': 10,
    'learning_rate': 0.01,
    'momentum': 0,
    'hidden_units': 100
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load trainings dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

# load test dataset
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=hyper_params['batch_size'],
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=hyper_params['test_batch_size'],
                                          shuffle=True)


def setup():
    pass


def train_network(filename='model.pt'):
    # create neural net and train (input is image, output is number between 0 and 9.
    model = NeuralNetwork(28 * 28, hyper_params['hidden_units'], 10)

    # Criterion and optimizer
    # might actually use MSE Error
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hyper_params['learning_rate'],
                                momentum=hyper_params['momentum'])

    # train and test the network
    for epoch in range(hyper_params['num_epochs']):
        train(train_loader, model, optimizer, criterion, epoch, hyper_params['num_epochs'])
        test(test_loader, model)

    # save the current model
    torch.save(model, './model/' + filename)
    print('Saved pretrained model to ./model/' + filename)


def prune_network(prune_strategy=None):
    # setup variables for pruning
    current_pruning_rate = hyper_params['pruning_percentage']
    s = pd.DataFrame(columns=['epoch', 'accuracy', 'pruning_perc'])

    while current_pruning_rate <= 100:
        # loading the model
        print('loading pre-trained model for pruning with pruning percentage of {:.4f} %'
              .format(current_pruning_rate))
        model = torch.load('./model/model.pt')
        model.train()

        # loss and optimizer for the loaded model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hyper_params['learning_rate'],
                                    momentum=hyper_params['momentum'])

        # prune using strategy
        strategy = PruneNeuralNetStrategy(prune_strategy)

        loss = None
        if strategy.requires_loss():
            print('Calculate network loss for 2nd order derivative')
            train_loader_second_order = torch.utils.data.DataLoader(train_dataset,
                                                                    batch_size=train_dataset.__len__())

            for (images, labels) in train_loader_second_order:
                images = images.reshape(-1, 28 * 28)
                outputs = model(images)
                loss = criterion(outputs, labels)

        strategy.prune(model, current_pruning_rate, loss=loss)

        # setup data frame for results
        accuracy = np.zeros(hyper_params['num_retrain_epochs'] + 1)

        # Reevaluate the network performance
        accuracy[0] = test(test_loader, model)

        # Retrain and reevaluate
        for epoch in range(hyper_params['num_retrain_epochs']):
            train(train_loader, model, optimizer, criterion, epoch,
                  hyper_params['num_retrain_epochs'])
            accuracy[epoch + 1] = test(test_loader, model)

        # accumulate data
        tmp = pd.DataFrame({'epoch': range(hyper_params['num_retrain_epochs'] + 1),
                            'accuracy': accuracy,
                            'pruning_perc': np.full(hyper_params['num_retrain_epochs'] + 1, current_pruning_rate / 100)})
        s = s.append(tmp, ignore_index=True)

        # update current pruning rate
        current_pruning_rate = current_pruning_rate + hyper_params['pruning_update_rate']

    # plot the results
    sns.set()
    sns.set_context("talk")
    plot = sns.relplot(x='epoch', y='accuracy', hue='pruning_perc', legend='full',
                       kind="line", data=s, linewidth=2)
    plt.show(plot)


# train_network()
# info: saved network's performance: 96.05 %
prune_network(magnitude_based_pruning)

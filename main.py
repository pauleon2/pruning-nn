import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pruning_nn.network import NeuralNetwork, get_single_pruning_layer, get_network_weight_count
from pruning_nn.pruning import PruneNeuralNetStrategy, magnitude_based_pruning, random_pruning, obd_pruning
from pruning_nn.util import train, test

# constant variables
hyper_params = {
    'pruning_percentage': 20,  # percentage of weights pruned
    'pruning_update_rate': 2,
    'batch_size': 64,
    'test_batch_size': 100,
    'num_retrain_epochs': 2,
    'num_epochs': 20,
    'learning_rate': 0.01,
    'momentum': 0,
    'hidden_units': 100
}

result_folder = './out/result/'
model_folder = './out/model/'
dataset_folder = './dataset/mnist'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load trainings dataset
train_dataset = torchvision.datasets.MNIST(root=dataset_folder,
                                           train=True,
                                           transform=transform,
                                           download=True)

# load test dataset
test_dataset = torchvision.datasets.MNIST(root=dataset_folder,
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
    if not os.path.exists('./out'):
        os.mkdir('./out')
    if not os.path.exists('./out/model'):
        os.mkdir('./out/model')
        print('Created directory for model')
    if not os.path.exists('./out/result'):
        os.mkdir('./out/result')
        print('Created directory for results')


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
    torch.save(model, model_folder + filename)
    print('Saved pre-trained model to ' + model_folder + filename)


def prune_network(prune_strategy=None, filename='model.pt', runs=1):
    # setup variables for pruning
    pruning_rate = hyper_params['pruning_percentage']

    # prune using strategy
    strategy = PruneNeuralNetStrategy(prune_strategy)

    # output variables
    out_name = result_folder + str(prune_strategy.__name__)
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method'])

    for i in range(runs):
        # load model
        model = torch.load(model_folder + filename)
        # loss and optimizer for the loaded model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hyper_params['learning_rate'],
                                    momentum=hyper_params['momentum'])

        while get_network_weight_count(model).item() > 100:
            print('Prune model with ' + str(get_network_weight_count(model).item()) + ' weights using ' + str(
                prune_strategy.__name__))

            loss = None
            if strategy.requires_loss():
                # todo: introduce cross validation set
                print('Calculate network loss for 2nd order derivative')
                train_loader_second_order = torch.utils.data.DataLoader(train_dataset,
                                                                        batch_size=train_dataset.__len__())

                for (images, labels) in train_loader_second_order:
                    images = images.reshape(-1, 28 * 28)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                print('Finished calculating loss for 2nd order derivative. The overall error is {:.4f}'
                      .format(loss.item()))

            strategy.prune(model, pruning_rate, loss=loss)

            # Retrain and reevaluate
            if strategy.require_retraining():
                for epoch in range(hyper_params['num_retrain_epochs']):
                    train(train_loader, model, optimizer, criterion, epoch,
                          hyper_params['num_retrain_epochs'])

            # test network performance after pruning and retraining
            accuracy = test(test_loader, model)

            # accumulate data
            tmp = pd.DataFrame({'run': [i],
                                'accuracy': [accuracy],
                                'pruning_perc': [pruning_rate],
                                'number_of_weights': [get_network_weight_count(model)],
                                'pruning_method': [str(prune_strategy.__name__)]})
            s = s.append(tmp, ignore_index=True)

        # save model
        torch.save(model, out_name + '-model' + str(i) + '.pt')

    # save data frame
    s.to_pickle(out_name + '.pkl')


# info: saved network's performance: 96.44 %
setup()
train_network()
for strat in [random_pruning, magnitude_based_pruning]:
    prune_network(prune_strategy=strat, runs=3)

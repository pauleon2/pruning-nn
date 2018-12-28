import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pruning_nn.network import NeuralNetwork, get_single_pruning_layer, get_network_weight_count
from pruning_nn.pruning import PruneNeuralNetStrategy, magnitude_based_pruning, random_pruning_uniform, random_pruning_uniform_abs, \
    obd_pruning, magnitude_based_pruning_abs, obd_pruning_abs
from pruning_nn.util import train, test
import logging

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
        logging.info('Created directory for model')
    if not os.path.exists('./out/result'):
        os.mkdir('./out/result')
        logging.info('Created directory for results')


def setup_training(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params['learning_rate'],
                                momentum=hyper_params['momentum'])
    return loss_func, optimizer


def calculate_2nd_order_loss(model, criterion):
    logging.info('Calculate network loss for 2nd order derivative')
    train_loader_second_order = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=train_dataset.__len__())
    # todo: introduce cross validation set
    loss = None
    for (images, labels) in train_loader_second_order:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        loss = criterion(outputs, labels)
    logging.info('Finished calculating loss for 2nd order derivative. The overall error is {:.4f}'
                 .format(loss.item()))
    return loss


def train_network(filename='model.pt'):
    # create neural net and train (input is image, output is number between 0 and 9.
    model = NeuralNetwork(28 * 28, hyper_params['hidden_units'], 10)

    # Criterion and optimizer
    # might actually use MSE Error
    criterion, optimizer = setup_training(model)

    # train and test the network
    for epoch in range(hyper_params['num_epochs']):
        train(train_loader, model, optimizer, criterion, epoch, hyper_params['num_epochs'])
        test(test_loader, model)

    # save the current model
    torch.save(model, model_folder + filename)
    logging.info('Saved pre-trained model to ' + model_folder + filename)


def prune_network(prune_strategy=None, filename='model.pt', runs=1):
    # setup variables for pruning
    pruning_rates = [10, 15, 25]  # experiment for 10 and 20 percent of the weights each step.

    # prune using strategy
    strategy = PruneNeuralNetStrategy(prune_strategy)

    # output variables
    out_name = result_folder + str(prune_strategy.__name__)
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method'])

    for rate in pruning_rates:

        for i in range(runs):
            # load model
            model = torch.load(model_folder + filename)
            # loss and optimizer for the loaded model
            criterion, optimizer = setup_training(model)

            while get_network_weight_count(model).item() > 500:
                logging.info(
                    'Prune model with ' + str(get_network_weight_count(model).item()) + ' weights using ' + str(
                        prune_strategy.__name__))

                loss = None
                if strategy.requires_loss():
                    loss = calculate_2nd_order_loss(model, criterion)  # calculate loss
                    torch.save(model, out_name + '.pt')  # save the model if server breaks
                    s.to_pickle(out_name + '.pkl')  # save data so far if server breaks

                strategy.prune(model, rate, loss=loss)

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
                                    'pruning_perc': [rate],
                                    'number_of_weights': [get_network_weight_count(model).item()],
                                    'pruning_method': [str(prune_strategy.__name__)]
                                    })
                s = s.append(tmp, ignore_index=True)

        # save data frame
        s.to_pickle(out_name + '.pkl')


def prune_network_abs(prune_strategy=None, filename='model.pt', runs=1):
    pruning_rates = [1000, 2000, 5000]

    # prune using strategy
    strategy = PruneNeuralNetStrategy(prune_strategy)

    out_name = result_folder + str(prune_strategy.__name__)
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_num', 'number_of_weights', 'pruning_method'])

    for rate in pruning_rates:
        for i in range(runs):
            model = torch.load(model_folder + filename)
            # loss and optimizer for the loaded model
            criterion, optimizer = setup_training(model)
            while get_network_weight_count(model).item() > rate:

                loss = None
                if strategy.requires_loss():
                    loss = calculate_2nd_order_loss(model, criterion)
                    torch.save(model, out_name + '.pt')  # save the model if server breaks
                    s.to_pickle(out_name + '.pkl')  # save data so far if server breaks

                strategy.prune(model, rate, loss=loss)

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
                                    'pruning_num': [rate],
                                    'number_of_weights': [get_network_weight_count(model).item()],
                                    'pruning_method': [str(prune_strategy.__name__)]
                                    })
                s = s.append(tmp, ignore_index=True)

    # save the data frame
    s.to_pickle(out_name + '.pkl')


if __name__ == '__main__':
    # setup environment
    setup()
    logging.basicConfig(filename='out/myapp.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # train the model
    train_network()

    # prune with percentage p
    for strat in [random_pruning_uniform, magnitude_based_pruning]:
        prune_network(prune_strategy=strat, runs=50)

    # prune absolute top k
    for strat in [random_pruning_uniform_abs, magnitude_based_pruning_abs]:
        prune_network_abs(prune_strategy=strat, runs=50)

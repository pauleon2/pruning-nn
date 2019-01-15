import os
import logging
import torch
import torch.nn as nn
import pandas as pd

from pruning_nn.network import NeuralNetwork, MultiLayerNeuralNetwork
from pruning_nn.util import get_network_weight_count
from pruning_nn.pruning import PruneNeuralNetStrategy, magnitude_class_blinded, magnitude_class_uniform, \
    random_pruning
from util.learning import train, test, cross_validation_error
from util.dataloader import get_train_valid_dataset, get_test_dataset

# constant variables
hyper_params = {
    'num_retrain_epochs': 2,
    'max_retrain_epochs': 5,
    'num_epochs': 20,
    'learning_rate': 0.01,
    'momentum': 0
}

result_folder = './out/result/'
model_folder = './out/model/'

test_set = get_test_dataset()
train_set, valid_set = get_train_valid_dataset()


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


def train_network(filename='model', multi_layer=False):
    # create neural net and train (input is image, output is number between 0 and 9.
    model = NeuralNetwork(28 * 28, 100, 10)
    if multi_layer:
        model = MultiLayerNeuralNetwork(28 * 28, 30, 10)

    # Criterion and optimizer
    # might actually use MSE Error
    criterion, optimizer = setup_training(model)

    # train and test the network
    for epoch in range(hyper_params['num_epochs']):
        train(train_set, model, optimizer, criterion)
        # test(test_loader, model)

    # save the current model
    torch.save(model, model_folder + filename + '.pt')
    logging.info('Saved pre-trained model to ' + model_folder + filename)


def prune_network(prune_strategy, filename='model', runs=1):
    # setup variables for pruning
    pruning_rates = [10, 15, 25]  # experiment for 10, 15 and 25 percent of the weights each step.

    # prune using strategy
    strategy = PruneNeuralNetStrategy(prune_strategy)

    # output variables
    out_name = result_folder + str(prune_strategy.__name__) + '-' + filename
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method'])

    # prune with different pruning rates
    for rate in pruning_rates:

        # repeat all experiments a fixed number of times
        for i in range(runs):
            # load model
            model = torch.load(model_folder + filename + '.pt')

            # loss and optimizer for the loaded model
            criterion, optimizer = setup_training(model)

            # prune as long as there are more than 500 elements in the network
            while get_network_weight_count(model).item() > 500:
                loss = None
                if strategy.requires_loss():  # todo: should probably done in another way
                    loss = cross_validation_error(valid_set, model, criterion)  # calculate loss
                    torch.save(model, out_name + '.pt')  # save the model if server breaks
                    s.to_pickle(out_name + '.pkl')  # save data so far if server breaks

                org_acc = test(test_set, model)
                strategy.prune(model, rate, loss=loss)

                # Retrain and reevaluate
                pruned_acc = test(test_set, model)
                if strategy.require_retraining():
                    # todo: use dynamic criterion instead e.g. the test accuracy drop
                    for epoch in range(hyper_params['num_retrain_epochs']):
                        train(train_set, model, optimizer, criterion)

                # test network performance after pruning and retraining
                # todo: save pruned model with best accuracy
                retrained_accuracy = test(test_set, model)

                # accumulate data
                tmp = pd.DataFrame({'run': [i],
                                    'accuracy': [retrained_accuracy],
                                    'pruning_perc': [rate],
                                    'number_of_weights': [get_network_weight_count(model).item()],
                                    'pruning_method': [str(prune_strategy.__name__)]
                                    })
                s = s.append(tmp, ignore_index=True)

        # save data frame
        s.to_pickle(out_name + '.pkl')


if __name__ == '__main__':
    # setup environment
    setup()
    logging.basicConfig(filename='out/myapp.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # train the model
    for j in range(8):
        name = 'model' + str(j)
        multi = j >= 4

        train_network(filename=name, multi_layer=multi)

        # prune with percentage p
        for strat in [random_pruning, magnitude_class_blinded, magnitude_class_uniform]:
            prune_network(prune_strategy=strat, filename=name, runs=25)

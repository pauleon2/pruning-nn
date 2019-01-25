import os
import logging
import time
import torch
import torch.nn as nn
import pandas as pd

from pruning_nn.network import NeuralNetwork, MultiLayerNeuralNetwork
from pruning_nn.util import get_network_weight_count
from pruning_nn.pruning import PruneNeuralNetStrategy, magnitude_class_blinded, magnitude_class_uniform, \
    random_pruning, optimal_brain_damage, optimal_brain_surgeon_layer_wise, net_trim
from util.learning import train, test, cross_validation_error
from util.dataloader import get_train_valid_dataset, get_test_dataset

# constant variables
hyper_params = {
    'num_retrain_epochs': 2,
    'max_retrain_epochs': 10,
    'num_epochs': 20,
    'learning_rate': 0.01,
    'momentum': 0
}

result_folder = './out/result/'
model_folder = './out/model/'

test_set = get_test_dataset()
train_set, valid_set = get_train_valid_dataset(valid_batch=100)


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


def prune_network(prune_strategy, filename='model', runs=1, save_models=False):
    # setup variables for pruning
    pruning_rates = [60, 50, 40]  # experiment for 10, 15 and 25 percent of the weights each step.

    # prune using strategy
    strategy = PruneNeuralNetStrategy(prune_strategy)
    if strategy.requires_loss():
        # if optimal brain damage is used get dataset with only one batch
        if prune_strategy in [optimal_brain_damage, net_trim]:
            btx = None
        else:
            btx = 100
        _, strategy.valid_dataset = get_train_valid_dataset(valid_batch=btx)
        strategy.criterion = nn.CrossEntropyLoss()  # todo: put in own method maybe

    # output variables
    out_name = result_folder + str(prune_strategy.__name__) + '-' + filename
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method'])

    # set variables for the best models with initial values
    best_acc = 0
    smallest_model = 30000

    # prune with different pruning rates
    for rate in pruning_rates:

        # repeat all experiments a fixed number of times
        for i in range(runs):
            # load model
            model = torch.load(model_folder + filename + '.pt')
            original_acc = test(test_set, model)
            original_weight_count = get_network_weight_count(model)

            # loss and optimizer for the loaded model
            criterion, optimizer = setup_training(model)

            # prune as long as there are more than 500 elements in the network
            while get_network_weight_count(model).item() > 500:
                start = time.time()
                strategy.prune(model, rate)

                # Retrain and reevaluate the process
                if strategy.require_retraining():
                    untrained_test_acc = test(test_set, model)
                    untrained_acc = test(valid_set, model)
                    # setup variables for loop retraining
                    prev_acc = untrained_acc
                    retrain = True
                    retrain_epoch = 1

                    while retrain:
                        train(train_set, model, optimizer, criterion)
                        new_acc = test(valid_set, model)

                        # stop retraining if the test accuracy imporves only slightly or the maximum number of
                        # retrainnig epochs is reached
                        if new_acc - prev_acc < 0.01 or retrain_epoch >= hyper_params['num_retrain_epochs']:
                            retrain = False
                        else:
                            retrain_epoch += 1
                            prev_acc = new_acc

                    final_acc = test(test_set, model)
                    retrain_change = final_acc - untrained_test_acc
                else:
                    retrain_epoch = 0
                    final_acc = test(test_set, model)
                    retrain_change = 0

                # Save the best models with the following criterion
                # 1. smallest weight count with max 1% accuracy drop from the original model.
                # 2. best performing model overall with at least a compression rate of 50%.
                if save_models and (
                        (original_acc - final_acc < 1 and get_network_weight_count(model) < smallest_model) or (
                        get_network_weight_count(model) <= original_weight_count / 2 and final_acc > best_acc)):
                    # set the values to the new ones
                    best_acc = final_acc if final_acc > best_acc else best_acc
                    model_size = int(get_network_weight_count(model))
                    smallest_model = model_size if model_size < smallest_model else smallest_model

                    # save the model
                    torch.save(model, out_name + '-rate{}-weight{}-per{}.pt'
                               .format(str(rate), str(model_size), str(final_acc)))

                # evaluate duration of process
                time_needed = time.time() - start

                # accumulate data
                # todo: optimize the data frame append
                tmp = pd.DataFrame({'run': [i],
                                    'accuracy': [final_acc],
                                    'pruning_perc': [rate],
                                    'number_of_weights': [get_network_weight_count(model).item()],
                                    'pruning_method': [str(prune_strategy.__name__)],
                                    'time': [time_needed],
                                    'retrain_change': [retrain_change],
                                    'retrain_epochs': [retrain_epoch]
                                    })
                s = s.append(tmp, ignore_index=True, sort=True)

        # save data frame
        s.to_pickle(out_name + '.pkl')


if __name__ == '__main__':
    # setup environment
    setup()
    logging.basicConfig(filename='out/myapp.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # train_network()
    prune_network(magnitude_class_blinded)

    # train the model
    for j in range(8):
        name = 'model' + str(j)
        multi = j >= 4

        train_network(filename=name, multi_layer=multi)

        # prune with percentage p
        for strat in [random_pruning, magnitude_class_blinded, magnitude_class_uniform]:
            prune_network(prune_strategy=strat, filename=name, runs=25)

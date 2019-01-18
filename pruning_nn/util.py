import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from pruning_nn.network import MaskedLinearLayer


"""
This file contains some utility functions to calculate hessian matrix and its inverse.

Adapted from https://github.com/csyhhu/L-OBS/blob/master/PyTorch/ImageNet/util.py
Author: Chen Shangyu (schen025@e.ntu.edu.sg)
"""


def generate_hessian_inverse_fc(layer, hessian_inverse_path, layer_input_train_dir):
    """
    This function calculate hessian inverse for a fully-connect layer
    :param hessian_inverse_path: the hessian inverse path you store
    :param layer: the layer weights
    :param layer_input_train_dir: layer inputs in the dir
    :return:
    """

    w_layer = layer.get_weight().data.numpy().T
    n_hidden_1 = w_layer.shape[0]

    # Here we use a recursive way to calculate hessian inverse
    hessian_inverse = 1000000 * np.eye(n_hidden_1 + 1)

    dataset_size = 0
    for input_index, input_file in enumerate(os.listdir(layer_input_train_dir)):
        layer2_input_train = np.load(layer_input_train_dir + '/' + input_file)

        if input_index == 0:
            dataset_size = layer2_input_train.shape[0] * len(os.listdir(layer_input_train_dir))

        for i in range(layer2_input_train.shape[0]):
            vect_w_b = np.vstack((np.array([layer2_input_train[i]]).T, np.array([[1.0]])))
            denominator = dataset_size + np.dot(np.dot(vect_w_b.T, hessian_inverse), vect_w_b)
            numerator = np.dot(np.dot(hessian_inverse, vect_w_b), np.dot(vect_w_b.T, hessian_inverse))
            hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)

        if input_index % 100 == 0:
            print('[%s] Finish processing batch %s' % (datetime.now(), input_index))

    np.save(hessian_inverse_path, hessian_inverse)
    print('[%s]Hessian Inverse Done!' % (datetime.now()))


def generate_hessian_inverse_fc(hessian_inverse_path, w_layer_path, layer_input_train_dir):
    """

    :param hessian_inverse_path:    Path where the hessian inverse should be deleted.
    :param w_layer_path:            The weight matrix of the layer.
    :param layer_input_train_dir:   The inputs on the layer.
    :return:
    """
    pass


"""
Normal util methods start here
"""


def prune_network_by_saliency(network, percentage):
    """
    Prune the number of percentage weights from the network. The elements are pruned according to the saliency that is
    set in the network. By default the saliency is the actual weight of the connections. The number of elements are
    pruned blinded. Meaning in each layer a different percentage of elements might get pruned but overall the given one
    is removed from the network. This is a different approach than used in the method below where we prune exactly the
    given percentage from each layer.

    :param network:     The layers of the network.
    :param percentage:  The percentage of weights that should be pruned.
    :return: The calculated threshold.
    """
    all_sal = []
    for layer in get_single_pruning_layer(network):
        # flatten both weights and mask
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())

        # zip, filter, unzip the two lists
        mask, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))
        # add all saliencies to list
        all_sal += filtered_saliency

    # calculate percentile
    th = np.percentile(np.array(all_sal), percentage)

    # set the mask
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.get_saliency(), th).float())


def prune_layer_by_saliency(network, percentage):
    for layer in get_single_pruning_layer(network):
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())
        mask, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))
        th = np.percentile(np.array(filtered_saliency), percentage)
        layer.set_mask(torch.ge(layer.get_saliency(), th).float())


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == MaskedLinearLayer:
            yield child


def get_single_pruning_layer_with_name(network):
    for name, child in network.named_children():
        if type(child) == MaskedLinearLayer:
            yield name, child


def get_weight_distribution(network):
    all_weights = []
    for layer in get_single_pruning_layer(network):
        mask = list(layer.get_mask().numpy().flatten())
        weights = list(layer.get_weight().data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))

        all_weights += list(filtered_weights)
    return pd.DataFrame(data=all_weights)


def get_network_weight_count(network):
    total_weights = 0
    for layer in get_single_pruning_layer(network):
        total_weights += layer.get_weight_count()
    return total_weights


def reset_pruned_network(network):
    for layer in get_single_pruning_layer(network):
        layer.reset_parameters(keep_mask=True)


def keep_input_layerwise(network):
    for layer in get_single_pruning_layer(network):
        layer.keep_layer_input = True

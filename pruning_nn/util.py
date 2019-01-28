import os
import torch
import numpy as np
import scipy.linalg as sla
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
            # todo: remove
            print('[%s] Finish processing batch %s' % (datetime.now(), input_index))

    np.save(hessian_inverse_path, hessian_inverse)
    # todo: remove
    print('[%s]Hessian Inverse Done!' % (datetime.now()))


def edge_cut(layer, hessian_inverse_path, cut_ratio):
    """
    This function prune weights of biases based on given hessian inverse and cut ratio
    :param hessian_inverse_path:
    :param layer:
    :param cut_ratio: The zeros percentage of weights and biases, or, 1 - compression ratio
    :return:
    """

    # dataset_size = layer2_input_train.shape[0]
    w_layer = layer.get_weight().data.numpy()

    # biases = layer.bias.data.numpy()
    n_hidden_1 = w_layer.shape[0]
    n_hidden_2 = w_layer.shape[1]

    sensitivity = np.array([])

    hessian_inverse = np.load(hessian_inverse_path)
    # todo: remove
    print('[%s] Hessian Inverse Done!' % datetime.now())

    gate_w = layer.get_mask().data.numpy()
    # gate_b = np.ones([n_hidden_2])

    # calculate number of pruneable elements
    max_pruned_num = int(layer.get_weight_count() * cut_ratio)  # todo: make sure this is exactly the same as percentage
    # todo: remove
    print('[%s] Max prune number : %d' % (datetime.now(), max_pruned_num))

    # Calcuate sensitivity score. Refer to Eq.5.
    for i in range(n_hidden_2):
        sensitivity = np.hstack(
            (sensitivity, 0.5 * ((w_layer.T[i] ** 2) / np.diag(hessian_inverse))))
    sorted_index = np.argsort(sensitivity)

    # todo: remove x2
    print('[%s] Sorted index generate completed.' % datetime.now())
    print('[%s] Starting Pruning!' % datetime.now())
    hessian_inverseT = hessian_inverse.T

    prune_count = 0
    for i in range(n_hidden_1 * n_hidden_2):
        prune_index = [sorted_index[i]]
        x_index = prune_index[0] / (n_hidden_1 + 1)  # next layer num
        y_index = prune_index[0] % (n_hidden_1 + 1)  # this layer num

        if gate_w[y_index][x_index] == 1:
            delta_w = (-w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]) * hessian_inverseT[y_index]
            gate_w[y_index][x_index] = 0
            prune_count += 1
            # Parameters update, refer to Eq.5
            w_layer.T[x_index] = w_layer.T[x_index] + delta_w[0:-1]
            # b_layer[x_index] = b_layer[x_index] + delta_w[-1]

        w_layer = w_layer * gate_w
        # b_layer = b_layer * gate_b

        if prune_count == max_pruned_num:
            # todo: remove
            print('[%s] Have prune required weights' % datetime.now())
            break

    # print 'Non-zeros: %d' %np.count_nonzero(w_layer)
    # print 'weights number: %d' %w_layer.size
    # todo: remove
    print('[%s] Prune Finish. compression ratio: %.3f' % (
        datetime.now(), 1 - (float(np.count_nonzero(w_layer)) / w_layer.size)))

    # set created mask to network again and update the weights
    layer.set_mask(torch.from_numpy(gate_w))
    layer.weight = torch.from_numpy(w_layer)

    # if not os.path.exists(prune_save_path):
    #    os.makedirs(prune_save_path)

    # np.save("%s/weights" % prune_save_path, w_layer)
    # np.save("%s/biases" % prune_save_path, b_layer)


"""
The following part solves the constraint problem for the Net Trim Algorithm.

Adapted from: https://dnntoolbox.github.io/Net-Trim/
arxiv: 1806.06457
"""


def net_trim_solver(X: np.ndarray, Y: np.ndarray, V, epsilon, rho, max_iteration=10000):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of data sample for X and Y are not the same.')

    if V is None:
        # use simple sparse least-squares solution
        return sparse_least_squares_solver(X, Y, epsilon, rho, max_iteration)

    # dimensions of signals
    N = X.shape[0]
    M = Y.shape[0]
    P = X.shape[1]

    C = np.matmul(X, X.transpose())
    # normalize the input matrices and parameters
    c0 = 0.3 * np.sqrt(C.max())
    C = C / c0 ** 2 + np.eye(C.shape[0])

    epsilon = epsilon / c0
    X = X / c0
    Y = Y / c0
    V = V / c0

    # Cholesky decomposition of C
    Lc = np.linalg.cholesky(C)
    Uc = Lc.transpose()

    Omega = np.where(Y > 1e-6)
    Omega_c = np.where(Y <= 1e-6)

    Y = Y[Omega]
    V = V[Omega_c]

    # initialization
    U1 = 0
    U2 = 0

    W1 = np.zeros(shape=(M, P))
    W3 = np.zeros(shape=(N, M))
    W2 = np.zeros(shape=(N, M))

    thr = M * N * 5e-9
    for cnt in range(max_iteration):
        Z = np.matmul(W3.transpose(), X) - U1

        W_prev = W3

        # a- updating W1[Omega]
        D = Z[Omega] - Y
        gain = epsilon / np.maximum(np.linalg.norm(D), epsilon)
        W1[Omega] = Y + gain * D

        # b- updating W1[Omega_c]
        W1[Omega_c] = Z[Omega_c] - np.maximum(Z[Omega_c] - V, 0)

        # c- updating W2
        D = W3 - U2
        W2 = np.sign(D) * np.maximum(np.abs(D) - 1 / rho, 0)

        # d- updating W3
        D = np.matmul(X, (W1 + U1).transpose()) + W2 + U2
        _1 = sla.solve_triangular(Lc, D, lower=True)
        W3 = sla.solve_triangular(Uc, _1, lower=False)

        # e- updating U1
        U1 = U1 + W1 - np.matmul(W3.transpose(), X)
        U2 = U2 + W2 - W3

        if np.linalg.norm(W3 - W_prev) < thr:
            break

        if cnt % 500 == 0:
            # todo: remove
            print('{0} : {1:3.6f}'.format(cnt, np.linalg.norm(W3 - W_prev)), flush=True)

    # ultimately, W2 and W3 would be the same
    return W2


def sparse_least_squares_solver(X: np.ndarray, Y: np.ndarray, epsilon, rho, max_iteration=10000):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of data sample for X and Y are not the same.')

    # dimensions of signals
    N = X.shape[0]
    M = Y.shape[0]

    C = np.matmul(X, X.transpose())
    # normalize the input matrices and parameters
    c0 = 0.3 * np.sqrt(C.max())
    C = C / c0 ** 2 + np.eye(C.shape[0])

    epsilon = epsilon / c0
    X = X / c0
    Y = Y / c0

    # Cholesky decomposition of C
    Lc = np.linalg.cholesky(C)
    Uc = Lc.transpose()

    # initialization
    U1 = 0
    U2 = 0

    W3 = np.zeros(shape=(N, M))
    W2 = np.zeros(shape=(N, M))

    thr = M * N * 5e-9
    for cnt in range(max_iteration):
        Z = np.matmul(W3.transpose(), X) - U1

        W_prev = W3

        # a- updating W1
        D = Z - Y
        gain = epsilon / np.maximum(np.linalg.norm(D), epsilon)
        W1 = Y + gain * D

        # b- updating W2
        D = W3 - U2
        W2 = np.sign(D) * np.maximum(np.abs(D) - 1 / rho, 0)

        # c- updating W3
        D = np.matmul(X, (W1 + U1).transpose()) + W2 + U2
        _1 = sla.solve_triangular(Lc, D, lower=True)
        W3 = sla.solve_triangular(Uc, _1, lower=False)

        # d- updating U1, U2
        U1 = U1 + W1 - np.matmul(W3.transpose(), X)
        U2 = U2 + W2 - W3

        if np.linalg.norm(W3 - W_prev) < thr:
            break

        if cnt % 500 == 0:
            # todo: remove
            print('{0} : {1:3.6f}'.format(cnt, np.linalg.norm(W3 - W_prev)), flush=True)

    # ultimately, W2 and W3 would be the same
    return W2


#
# My own utility methods start here
#
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
    # calculate the network's threshold
    th = find_network_threshold(network, percentage)

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


def find_network_threshold(network, percentage):
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
    return np.percentile(np.array(all_sal), percentage)


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == MaskedLinearLayer:
            yield child


def get_layer_count(network):
    i = 0
    for _ in get_single_pruning_layer_with_name(network):
        i += 1
    return i


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

    # return all the weights, that are not masked as a numpy array
    return np.array(all_weights)


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

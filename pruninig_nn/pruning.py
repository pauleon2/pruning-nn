from pruninig_nn.network import PruningLayer
import numpy as np
import random
import torch


class PruneNeuralNetStrategy:
    """
    Strategy pattern for the selection of the currently used pruning strategy.
    Strategies can be set during creation of the strategy object.
    """

    def __init__(self, strategy=None):
        """
        Creates a new PruneNeuralNetStrategy object. There are a number of pruning strategies supported currently there
        is random pruning only.

        :param strategy:    The selected strategy for pruning If no pruning strategy is provided random pruning will be
                            executed
        """
        if strategy:
            self.prune = strategy
        else:
            self.prune = random_pruning

    def prune(self, network, percentage):
        """
        :param: network: The network that should be pruned.
        :param: percentage: The percentage of weights that should be pruned.
        :return: The pruned network
        """
        self.prune(network, percentage)


def random_pruning(network, percentage):
    """
    Implementation of the random pruning. For each layer the given percentage of not yet pruned weights will be
    eliminated.
    :param network: The network where the pruning takes place.
    :param percentage: The percentage of weights that should be pruned.
    """
    for child in get_single_pruning_layer(network):
        mask = child.get_mask()
        total = mask.sum()  # All parameters that are non zero can be pruned
        prune_goal = percentage * total
        print('Total parameters {}, to be pruned {}'
              .format(total, prune_goal))
        prune_done = 0

        while prune_done < prune_goal:
            x = random.randint(0, child.wrapped.weight.size()[0] - 1)
            y = random.randint(0, child.wrapped.weight.size()[1] - 1)
            if mask[x][y] == 1:
                mask[x][y] = 0
                prune_done += 1
            else:
                continue
        child.set_mask(mask)


def weight_based_pruning(network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.
    :param network: The network where the pruning should be done.
    :param percentage: The percentage of not yet pruned weights that should be deleted.
    """
    layers = get_single_pruning_layer(network)
    threshold = find_threshold(layers=layers, percentage=percentage)
    for layer in get_single_pruning_layer(network):
        layer.set_mask(torch.ge(layer.wrapped.weight.data.abs(), threshold).float())
        # todo: second pruning run should only be allowed to prune not yet pruned values


def find_threshold(layers, percentage):
    """
    Find a threshold for a percentage so that the percentage number of values are below the threshold and the rest
    above.
    :param layers: The layers of the network.
    :param percentage: The percentage of weights that should be pruned.
    :return: The calculated threshold.
    """
    all_weights = []
    for layer in layers:
        # todo only add to all weights if weight is not masked yet.
        all_weights += list(filter(lambda x: x, layer.wrapped.weight.data.abs().numpy().flatten()))
    return np.percentile(np.array(all_weights), percentage * 100)


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == PruningLayer:
            yield child

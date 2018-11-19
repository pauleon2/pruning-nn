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
            self.prune_strategy = strategy
        else:
            self.prune_strategy = random_pruning

    def prune(self, network, percentage, loss=None):
        if self.requires_loss():
            self.prune_strategy(network, percentage, loss)
        else:
            self.prune_strategy(network, percentage)

    def requires_loss(self):
        """
        Check if the current pruning method needs retraining
        :return: If a gradient of the network is required.
        """
        return not (self.prune_strategy == random_pruning or self.prune_strategy == weight_based_pruning)


#
# Top-Down Pruning Approaches
#


def obd_pruning(network, percentage, loss):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param network: The network where the calculations should be done.
    :param percentage: The percentage of weights that should be pruned.
    :param loss: The loss of the network on the trainings set. Needs to have grad enabled.
    """
    params = network.parameters()
    grads = torch.autograd.grad(loss, params, create_graph=True)  # First order derivative

    for layer in get_single_pruning_layer(network):
        # step1: compute diagonal of hessian matrix
        hessian_diagonal = torch.Tensor(layer.wrapped.weight)  # TODO this step

        # step2: take weight matrix and square it
        weight_squared = torch.Tensor(layer.wrapped.weight ^ 2)

        # step3: multiply weight matrix with diagonal matrix
        # step4: divide by two
        saliency = (hessian_diagonal * weight_squared) * 0.5

        # make similar to weight based pruning


def obs_pruning(network, percentage, loss):
    """
    Implementation of the optimal brain surgeon algorithm.
    Requires the graidient to be set.

    :param network: The network that should be pruned
    :param percentage: The percentage of weights that should be pruned
    """
    pass


#
# Network based pruning methods
#


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
        prune_goal = (percentage * total) / 100
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


#
# UTIL METHODS
#


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
    return np.percentile(np.array(all_weights), percentage)


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == PruningLayer:
            yield child

import numpy as np
import random
import torch
from pruninig_nn.network import get_single_pruning_layer


class PruneNeuralNetStrategy:
    """
    Strategy pattern for the selection of the currently used pruning strategy.
    Strategies can be set during creation of the strategy object.
    Valid Strategies are:

    <ul>
        <li>Random Pruning</li>
        <li>Magnitude Based Pruning</li>
        <li>Optimal Brain Damage</li>
        <li>Optimal Brain Surgeon</li>
        <li>Net-Trim</li>
        <li>Layer-wise Optimal Brain Surgeon</li>
    </ul>

    All methods except of the random pruning and magnitude based pruning require the loss argument. In order to
    calculate the weight saliency in a top-down approach.
    If no Strategy is specified random pruning will be used as a fallback.
    """

    def __init__(self, strategy=None):
        """
        Creates a new PruneNeuralNetStrategy object. There are a number of pruning strategies supported currently there
        is random pruning only.

        :param strategy:    The selected strategy for pruning If no pruning strategy is provided random pruning will be
                            selected as the standard pruning method.
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
        Check if the current pruning method needs the network's loss as an argument.
        :return: If a gradient of the network is required.
        """
        return not (self.prune_strategy == random_pruning or self.prune_strategy == magnitude_based_pruning)

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: If the retraining is required.
        """
        return self.prune_strategy in [random_pruning, magnitude_based_pruning, obd_pruning]


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
        # todo step1: compute diagonal of hessian matrix
        hessian_diagonal = torch.Tensor(layer.wrapped.weight)

        # step2: take weight matrix and square it
        weight_squared = torch.Tensor(layer.wrapped.weight ** 2)

        # step3: multiply weight matrix with diagonal matrix
        # step4: divide by two can probably be skipped since it is a linear factor that does not effect the overall per.
        saliency = (hessian_diagonal * weight_squared) * 0.5

        # todo: make similar to magnitude based pruning


def obs_pruning(network, percentage, loss):
    """
    Implementation of the optimal brain surgeon algorithm.
    Requires the graidient to be set.

    :param network: The network that should be pruned
    :param percentage: The percentage of weights that should be pruned
    """
    pass


def layer_wise_obs_pruning(network, percentage, loss):
    pass


def net_trim_pruning(network, percentage, loss):
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
            # select random input and output node
            x = random.randint(0, child.wrapped.weight.size()[0] - 1)
            y = random.randint(0, child.wrapped.weight.size()[1] - 1)

            # if selected weight is still already pruned do nothing else prune this weight
            if mask[x][y] == 1:
                mask[x][y] = 0
                prune_done += 1
            else:
                continue
        child.set_mask(mask)


def magnitude_based_pruning(network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.
    :param network: The network where the pruning should be done.
    :param percentage: The percentage of not yet pruned weights that should be deleted.
    """
    layers = get_single_pruning_layer(network)
    threshold = find_threshold(layers=layers, percentage=percentage)
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.wrapped.weight.data.abs(), threshold).float())


#
# UTIL METHODS
#


def find_threshold(layers, percentage):
    """
    Find a threshold for a percentage so that the percentage number of values are below the threshold and the rest
    above. The threshold is always a positive number since this method uses only the absolute numbers of the weight
    magnitudes.
    :param layers: The layers of the network.
    :param percentage: The percentage of weights that should be pruned.
    :return: The calculated threshold.
    """
    all_weights = []
    for layer in layers:
        # flatten both weights and mask
        mask = list(layer.get_mask().abs().numpy().flatten())
        weights = list(layer.wrapped.weight.data.abs().numpy().flatten())  # TODO: check if these two are determinsitic

        # zip, filter, unzip the two lists
        mask, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))
        all_weights += filtered_weights

    return np.percentile(np.array(all_weights), percentage)

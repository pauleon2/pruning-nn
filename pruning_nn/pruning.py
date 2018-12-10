import numpy as np
import random
import torch
from torch.autograd import grad
from pruning_nn.network import get_single_pruning_layer


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
    # instead of using loss.backward(), use torch.autograd.grad() to compute 1st order gradients
    print('Start calculating first order derivative')
    loss_grads = grad(loss, network.parameters(), create_graph=True)
    print('Finished calculating first order derivative')

    # iterate over all parameter groups from the network
    for grd, layer in zip(loss_grads, get_single_pruning_layer(network)):
        print('Start calculating second order derivative for layer')
        all_grads = torch.ones(layer.get_weight().view(-1).size())

        for num, g in enumerate(grd.view(-1)):
            drv = grad(g, layer.get_weight(), retain_graph=True)
            all_grads[num] = drv[0].view(-1)[num]
            if (num + 1) % 100 == 0:
                print('Calculated 100 derivatives. Going on...')

        print('Finished calculating second order derivative for layer')

        with torch.no_grad:
            layer.grad = all_grads.view(layer.get_weight().size()) * layer.get_weight().data.pow(2) * 0.5

    # calculate threshold for pruning
    layers = get_single_pruning_layer(network)
    threshold = find_threshold(layers=layers, percentage=percentage)

    # update mask
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.get_weight().data.abs(), threshold).float())


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
            x = random.randint(0, child.get_weight().size()[0] - 1)
            y = random.randint(0, child.get_weight().size()[1] - 1)

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
        layer.set_mask(torch.ge(layer.get_weight().data.abs(), threshold).float())


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
    all_sal = []
    for layer in layers:
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

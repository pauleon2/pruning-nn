import numpy as np
import random
import torch
from torch.autograd import grad
from pruning_nn.network import get_single_pruning_layer, get_network_weight_count
import logging


class PruneNeuralNetStrategy:
    """
    Strategy pattern for the selection of the currently used pruning strategy.
    Strategies can be set during creation of the strategy object.
    Valid Strategies are:

    <ul>
        <li>Random Pruning</li>
        <li>Magnitude Based Pruning - Class blinded</li>
        <li>Magnitude Based Pruning - Uniform</li>
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

    def prune(self, network, value, loss=None):
        if self.requires_loss():
            self.prune_strategy(network, value, loss)
        else:
            self.prune_strategy(network, value)

    def requires_loss(self):
        """
        Check if the current pruning method needs the network's loss as an argument.
        :return: If a gradient of the network is required.
        """
        return self.prune_strategy in [obd_pruning_abs, obd_pruning]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: If the retraining is required.
        """
        return self.prune_strategy in [random_pruning, random_pruning_abs, magnitude_based_blinded,
                                       magnitude_based_blinded_abs, magnitude_based_uniform,
                                       magnitude_based_uniform_abs, obd_pruning, obd_pruning_abs]


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
    calculate_obd_saliency(network, loss)
    # calculate threshold for pruning
    threshold = find_network_threshold(network, percentage)
    prune_le_threshold(network, threshold)


def obd_pruning_abs(network, number_of_weights, loss):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param network: The network where the calculations should be done.
    :param number_of_weights: The total number of weights that should be pruned.
    :param loss: The loss of the network on the trainings set. Needs to have grad enabled.
    """
    calculate_obd_saliency(network, loss)
    # calculate threshold for pruning
    percentage = ((number_of_weights * 100) / get_network_weight_count(network)).numpy()
    threshold = find_network_threshold(network, percentage)
    prune_le_threshold(network, threshold)


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


def random_pruning_abs(network, number_of_weights):
    ratio = (number_of_weights * 100) / get_network_weight_count(network)
    random_pruning(network, ratio)


def magnitude_based_blinded(network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.
    :param network: The network where the pruning should be done.
    :param percentage: The percentage of not yet pruned weights that should be deleted.
    """
    threshold = find_network_threshold(network, percentage)
    prune_le_threshold(network, threshold)


def magnitude_based_blinded_abs(network, number_of_weights):
    percentage = ((number_of_weights * 100) / get_network_weight_count(network)).numpy()
    magnitude_based_blinded(network, percentage)


def magnitude_based_uniform(network, percentage):
    """
    Class blinded implementation of magnitude based pruning. There will be deleted the percentage of elements from each
    layer. In each layer the elements with the smallest magnitude will be pruned first
    :param network:
    :param percentage:
    :return:
    """
    for layer in get_single_pruning_layer(network):
        # zip, filter, unzip the two lists
        mask = list(layer.get_mask().abs().numpy().flatten())
        weight = list(layer.get_saliency().numpy().flatten())
        mask, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weight) if masked_val == 1))

        flattened_weights = np.array(filtered_weights)
        threshold = np.percentile(flattened_weights, percentage)
        layer.set_mask(torch.ge(layer.get_saliency(), threshold).float())


def magnitude_based_uniform_abs(network, number_of_weights):
    ratio = ((number_of_weights * 100) / get_network_weight_count(network)).numpy()
    magnitude_based_uniform(network, ratio)


#
# UTIL METHODS
#
def find_network_threshold(network, percentage):
    """
    Find a threshold for a percentage so that the percentage number of values are below the threshold and the rest
    above. The threshold is always a positive number since this method uses only the absolute numbers of the weight
    magnitudes.
    :param network: The layers of the network.
    :param percentage: The percentage of weights that should be pruned.
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
    return np.percentile(np.array(all_sal), percentage)


def prune_le_threshold(network, threshold):
    """
    Delete all elements from the network that are less than the threshold
    :return:
    """
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.get_saliency(), threshold).float())


def prune_ge_threshold(network, threshold):
    """
    Delete all elements that are greater than the threshold
    """
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.le(layer.get_saliency(), threshold).float())


def calculate_obd_saliency(network, loss):
    logging.info('Start obd procedure')
    weight_params = map(lambda x: x.get_weight(), get_single_pruning_layer(network))
    loss_grads = grad(loss, weight_params, create_graph=True)

    # iterate over all parameter groups from the network
    for grd, layer in zip(loss_grads, get_single_pruning_layer(network)):
        all_grads = []
        mask = layer.get_mask().view(-1)

        for num, (g, m) in enumerate(zip(grd.view(-1), mask)):
            if m.item() == 0:
                all_grads += [0]
            else:
                drv = grad(g, layer.get_weight(), retain_graph=True)
                all_grads += [drv[0].view(-1)[num].item()]
            if (num + 1) % 1000 == 0:
                logging.info('Calculated 1000 2nd order derivatives to be continued...')

        # set saliency
        layer.set_saliency(
            torch.tensor(all_grads).view(layer.get_weight().size()) * layer.get_weight().data.pow(2) * 0.5)

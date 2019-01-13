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
        <li>Magnitude Pruning Blinded</li>
        <li>Magnitude Pruning Uniform</li>
        <li>Optimal Brain Damage</li>
    </ul>

    The following strategies will e implemented in the future

    <ul>
        <li>Optimal Brain Surgeon</li>
        <li>Layer-wise Optimal Brain Surgeon</li>
    </ul>

    All methods except of the random pruning and magnitude based pruning require the loss argument. In order to
    calculate the weight saliency in a top-down approach.
    If no Strategy is specified random pruning will be used as a fallback.
    """

    def __init__(self, strategy):
        """
        Creates a new PruneNeuralNetStrategy object. There are a number of pruning strategies supported currently there
        is random pruning only.

        :param strategy:    The selected strategy for pruning If no pruning strategy is provided random pruning will be
                            selected as the standard pruning method.
        """
        self.prune_strategy = strategy

    def prune(self, network, value, loss=None):
        if self.requires_loss():
            self.prune_strategy(network, value, loss)
        else:
            self.prune_strategy(network, value)

    def requires_loss(self):
        """
        Check if the current pruning method needs the network's loss as an argument.
        :return: True iff a gradient of the network is required.
        """
        return self.prune_strategy in [optimal_brain_damage, gradient_magnitude]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: True iff the retraining is required.
        """
        return self.prune_strategy not in []


#
# Top-Down Pruning Approaches
#
def optimal_brain_damage(network, percentage, loss):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param network:     The network where the calculations should be done.
    :param percentage:  The percentage of weights that should be pruned.
    :param loss:        The loss of the network on the trainings set. Needs to have grad enabled.
    """
    calculate_obd_saliency(network, loss)
    prune_network_by_saliency(network, percentage)


def optimal_brain_surgeon(network, percentage, loss):
    pass


def optimal_brain_surgeon_layer_wise(network, percentage, loss):
    pass


#
# Random pruning
#
def random_pruning(network, percentage):
    # set saliency to random values
    for layer in get_single_pruning_layer(network):
        layer.set_saliency(torch.rand_like(layer.get_weight()) * layer.get_mask())

    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)
    print(get_network_weight_count(network))


#
# Magnitude based approaches
#
def magnitude_class_blinded(network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.

    The here used method is the class blinded method mentioned in the paper by See et.al from 2016 (DOI: 1606.09274v1).
    The method is also known from the paper by Bachor et.al from 2018 where it was named the PruNet pruning technique
    (DOI: 10.1109/IJCNN.2018.8489764)

    :param network:     The network where the pruning should be done.
    :param percentage:  The percentage of not yet pruned weights that should be deleted.
    """
    prune_network_by_saliency(network, percentage)


def magnitude_class_uniform(network, percentage):
    prune_layer_by_saliency(network, percentage)


#
# Gradient based pruning
#
def gradient_magnitude(network, percentage, loss):
    pass


#
# UTIL METHODS
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

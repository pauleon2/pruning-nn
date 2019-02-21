import torch
import os
import numpy as np
from pruning_nn.util import get_single_pruning_layer, get_network_weight_count, prune_layer_by_saliency, \
    prune_network_by_saliency, generate_hessian_inverse_fc, edge_cut, keep_input_layerwise, \
    get_single_pruning_layer_with_name, get_layer_count, get_weight_distribution, \
    find_network_threshold, get_filtered_saliency, calculate_obd_saliency, set_random_saliency, set_distributed_saliency


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

    The following algorithms are currently under construction:
    <ul>
        <li>Layer-wise Optimal Brain Surgeon</li>
    </ul>

    Method that were considered but due to inefficiency not implemented:
    <ul>
        <li>Optimal Brain Surgeon</li>
        <li>Net-Trim</li>
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

        # dataset and loss function for error calculation
        self.criterion = None
        self.valid_dataset = None

    def prune(self, network, value):
        """
        Wrapper method which calls the actual pruning strategy and computes how long it takes to complete the pruning
        step.

        :param network:     The network that should be pruned
        :param value:       The percentage of elements that should be pruned
        """
        self.prune_strategy(self, network, value)

    def requires_loss(self):
        """
        Check if the current pruning method needs the network's loss as an argument.
        :return: True iff a gradient of the network is required.
        """
        return self.prune_strategy in [optimal_brain_damage, optimal_brain_surgeon_layer_wise]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: True iff the retraining is required.
        """
        return self.prune_strategy not in [optimal_brain_surgeon_layer_wise]


#
# Top-Down Pruning Approaches
#
def optimal_brain_damage(self, network, percentage):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param self:        The strategy pattern object.
    :param network:     The network where the calculations should be done.
    :param percentage:  The percentage of weights that should be pruned.
    """
    # calculate the saliencies for the weights
    calculate_obd_saliency(self, network)
    # prune the elements with the lowest saliency in the network
    prune_network_by_saliency(network, percentage)


def optimal_brain_damage_absolute(self, network, number):
    calculate_obd_saliency(self, network)
    prune_network_by_saliency(network, number, percentage=False)


#
# Layer-wise approaches
#
def optimal_brain_surgeon_layer_wise(self, network, percentage):
    """
    Layer-wise calculation of the inverse of the hessian matrix. Then the weights are ranked similar to the original
    optimal brian surgeon algorithm.

    :param network:     The network that should be pruned.
    :param percentage:  What percentage of the weights should be pruned.
    :param self:        The strategy pattern object the method is attached to.
    """
    out_dir = './out/hessian'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_dir + '/layerinput')
        os.mkdir(out_dir + '/inverse')

    # where to put the cached layer inputs
    layer_input_path = out_dir + '/layerinput/'
    # where to save the hessian matricies
    hessian_inverse_path = out_dir + '/inverse/'

    # generate the input in the layers and save it for every batch
    keep_input_layerwise(network)

    for i, (images, labels) in enumerate(self.valid_dataset):
        images = images.reshape(-1, 28 * 28)
        network(images)
        for name, layer in get_single_pruning_layer_with_name(network):
            layer_input = layer.layer_input.data.numpy()
            path = layer_input_path + name + '/'
            if not os.path.exists(path):
                os.mkdir(path)

            np.save(path + 'layerinput-' + str(i), layer_input)

    # generate the hessian matrix for each layer
    for name, layer in get_single_pruning_layer_with_name(network):
        hessian_inverse_location = hessian_inverse_path + name
        generate_hessian_inverse_fc(layer, hessian_inverse_location, layer_input_path + name)
    # prune the elements from the matrix
    # todo: evaluate if this can be done in upper for-loop

    for name, layer in get_single_pruning_layer_with_name(network):
        edge_cut(layer, hessian_inverse_path + name + '.npy', percentage)


#
# Random pruning
#
def random_pruning(self, network, percentage):
    set_random_saliency(network)
    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)


def random_pruning_absolute(self, network, number):
    set_random_saliency(network)
    prune_network_by_saliency(network, number, percentage=False)


#
# Magnitude based approaches
#
def magnitude_class_blinded(self, network, percentage):
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


def magnitude_class_blinded_absolute(self, network, number):
    prune_network_by_saliency(network, number, percentage=False)


def magnitude_class_uniform(self, network, percentage):
    prune_layer_by_saliency(network, percentage)


def magnitude_class_uniform_absolute(self, network, number):
    prune_layer_by_saliency(network, number, percentage=False)


def magnitude_class_distributed(self, network, percentage):
    """
    This idea comes from the paper 'Learning both Weights and Connections for Efficient Neural Networks'
    (arXiv:1506.02626v3). The main idea is that in each layer respectively to the standard derivation many elements
    should be deleted.
    For each layer prune the weights w for which the following holds:

    std(layer weights) * t > w      This is equal to the following
    t > w/std(layer_weights)        Since std is e(x - e(x))^2 and as square number positive.

    So all elements for which the wright divided by the std. derivation is smaller than some threshold will get deleted

    :param network:     The network that should be pruned.
    :param percentage:  The number of elements that should be pruned.
    :return:
    """
    # set saliency
    set_distributed_saliency(network)
    # prune network
    prune_network_by_saliency(network, percentage)


def magnitude_class_distributed_absolute(self, network, number):
    set_distributed_saliency(network)
    prune_network_by_saliency(network, number, percentage=False)

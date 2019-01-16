import torch
from torch.autograd import grad
from pruning_nn.util import get_single_pruning_layer, get_network_weight_count, prune_layer_by_saliency, \
    prune_network_by_saliency
from util.learning import cross_validation_error


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

    Optionally the following will be implemented:
    <ul>
        <li>Magnitude Pruning Distribution</li>
        <li>Gradient Magnitude Pruning</li>
    </ul

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
        return self.prune_strategy in [optimal_brain_damage, gradient_magnitude, optimal_brain_surgeon,
                                       optimal_brain_surgeon_layer_wise]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: True iff the retraining is required.
        """
        return self.prune_strategy not in [optimal_brain_surgeon, optimal_brain_surgeon_layer_wise]


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
    # the loss of the network on the cross validation set
    loss = cross_validation_error(self.valid_dataset, network, self.criterion)
    print(loss)

    # Use GPU optimization if available
    if torch.cuda.is_available():
        network.cuda()
        loss.cuda()

    # calculate the first order gradients for all weights from the pruning layers.
    weight_params = map(lambda x: x.get_weight(), get_single_pruning_layer(network))
    loss_grads = grad(loss, weight_params, create_graph=True)

    # iterate over all layers and zip them with their corrosponding first gradient
    for grd, layer in zip(loss_grads, get_single_pruning_layer(network)):
        all_grads = []
        mask = layer.get_mask().view(-1)
        weight = layer.get_weight()

        # zip gradient and mask of the network in a lineared fashion
        for num, (g, m) in enumerate(zip(grd.view(-1), mask)):
            if m.item() == 0:
                # if the element is pruned i.e. if mask == 0 then the second order derivative should e zero as well
                # so no computations are needed
                all_grads += [0]
            else:
                # create the second order derivative and add it to the list which contains all gradients
                drv = grad(g, weight, retain_graph=True)
                all_grads += [drv[0].view(-1)[num].item()]

        # rearrange calculated value to their normal form and set saliency
        layer.set_saliency(
            torch.tensor(all_grads).view(weight.size()) * layer.get_weight().data.pow(2) * 0.5)

    # prune the elements with the lowest saliency in the network
    prune_network_by_saliency(network, percentage)


def optimal_brain_surgeon(self, network, percentage):
    # enable keeping of original inputs of the network
    network.keep_layer_input = True

    # calculate the loss of the network
    loss = cross_validation_error(self.valid_dataset, network, self.criterion)

    prune_goal = (get_network_weight_count(network) * percentage) / 100
    prune_done = 0
    print(prune_goal)

    while prune_goal < prune_done:
        # todo: implement obs
        pass


def optimal_brain_surgeon_layer_wise(self, network, percentage):
    """
    Layer-wise calculation of the inverse of the hessian matrix. Then the weights are ranked similar to the original
    optimal brian surgeon algorithm.

    IMPORTANT: Since no retraining is required it is not important how many elements are pruned in each step. Since for
    every single pruning element the complete inverse of the hessian will e calculated.

    :param network:     The network that should be pruned.
    :param percentage:  What percentage of the weights should be pruned.
    :param self:        The strategy pattern object the method is attached to.
    """
    prune_goal = (get_network_weight_count(network) * percentage) / 100
    prune_done = 0
    print(prune_goal)
    # todo: implement
    exit()

    """
    # calculate the hessian inverse for every layer and save it to a file
    for layer in get_single_pruning_layer(network):
        layer_name = layer.__name__
        n_hessian_batch_used = 1
        hessian_batch_size = 1
        use_Woodbury_tfbackend = True
        stride_factor = 1
        # todo: loss should be a parameter of ths function
        hessian_inv = generate_hessian_inv_woodbury(net=network,
                                                    trainloader=self.valid_dataset,
                                                    layer_name=layer_name,
                                                    layer_type='F',
                                                    n_batch_used=n_hessian_batch_used,
                                                    batch_size=hessian_batch_size,
                                                    use_tf_backend=use_Woodbury_tfbackend,
                                                    stride_factor=stride_factor,
                                                    use_cuda=False)

        np.save('%s/%s.npy' % ('./out/hessian/', layer_name), hessian_inv)

        weight = layer.get_weight().data.numpy()
        bias = layer.bias.data.numpy()
        wb = np.hstack([weight, bias.reshape(-1, 1)]).transpose()
        l1, l2 = wb.shape

        L = np.zeros([l1 * l2])
        for row_idx in range(l1):
            for col_idx in range(l2):
                L[row_idx * l2 + col_idx] = np.power(wb[row_idx, col_idx], 2) / (hessian_inv[row_idx, row_idx] + 10e-6)

        sen_rank = np.argsort(L)
        n_prune = l1 * l2
        mask = layer.get_mask().numpy()

        for i in range(n_prune):
            prune_idx = sen_rank[i]
            prune_row_idx = prune_idx / l2
            prune_col_idx = prune_idx % l2
            delta_W = - wb[prune_row_idx, prune_col_idx] / (
                    hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
            wb[:, prune_col_idx] += delta_W
            mask[prune_row_idx, prune_col_idx] = 0
            # wb = np.multiply(wb, mask)
    """


#
# Random pruning
#
def random_pruning(self, network, percentage):
    # set saliency to random values
    for layer in get_single_pruning_layer(network):
        layer.set_saliency(torch.rand_like(layer.get_weight()) * layer.get_mask())

    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)
    print(get_network_weight_count(network))


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


def magnitude_class_uniform(self, network, percentage):
    prune_layer_by_saliency(network, percentage)


def magnitude_class_distributed(self, network, percentage):
    """
    This idea comes from the paper 'Learning both Weights and Connections for Efficient Neural Networks'
    (arXiv:1506.02626v3). The main idea is that in each layer respectively to the standard derivation many elements
    should be deleted.

    :param network:     The network that should be pruned.
    :param percentage:  The number of elements that should be pruned.
    :return:
    """
    t = 0  # todo: the real problem here is how to determine the value t
    for layer in get_single_pruning_layer(network):
        th = layer.get_weight().data.std() * t
        layer.set_mask(torch.ge(layer.get_saliency(), th).float())


#
# Gradient based pruning
#
def gradient_magnitude(self, network, percentage):
    """
    Saliency measure based on the first order derivative i.e. the gradient of the weight. This idea is adapted from the
    paper 'Pruning CNN for Resource Efficient Interference' by Molchanov et.al who did this sort of pruning for
    convolutional neural networks(arXiv:1611.06440v2).
    The main idea is to delete weights with a small magnitude and a proportionally high gradient. Meaning their slope is
    relatively high.

    :param network:     The network that should be pruned.
    :param percentage:  The percentage of elements that should be pruned from the network.
    :param loss:        The overall loss of the network on the cross-validation set.
    """
    pass

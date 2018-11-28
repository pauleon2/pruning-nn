import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Feed-forward neural network with one hidden layer. The single layers are Prunable linear layers. In these single
    neorons or weights can be deleted and will therefore not be usable any more.
    Activation function: ReLU
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = PruningLayer(nn.Linear(input_size, hidden_size))
        self.relu = nn.ReLU()
        self.fc2 = PruningLayer(nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)


class PruningLayer(nn.Module):
    """
    Pruning Layer is a Decorator for all nn.Modules with a named parameter `weight`.
    The parameter 'weight' will be changed by calling the prune method. During the forwarding of data through the
    network the pruning matrix will be multiplied to the weights. This leads to a pruned layer since the weights for
    which the matrix holds a zero will be eliminated in the backproporgation algorithm.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        # create a mask of ones for all weights (no element pruned at beginning)
        self.mask = torch.ones(self.wrapped.weight.size())

    def get_mask(self):
        return self.mask

    def set_mask(self, mask):
        self.mask = mask

    def get_masked_weight(self):
        mask = list(self.mask.abs().numpy().flatten())
        weights = list(self.wrapped.weight.data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))
        return list(filtered_weights)

    def forward(self, x):
        self.wrapped.weight.data = self.wrapped.weight.data * self.mask
        return self.wrapped.forward(x)


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == PruningLayer:
            yield child

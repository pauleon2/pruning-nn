import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd


class NeuralNetwork(nn.Module):
    """
    Feed-forward neural network with one hidden layer. The single layers are Prunable linear layers. In these single
    neorons or weights can be deleted and will therefore not be usable any more.
    Activation function: ReLU
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = MaskedLinearLayer(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = MaskedLinearLayer(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)


class MaskedLinearLayer(nn.Linear):
    def __init__(self, in_feature, out_features, bias=False):
        super().__init__(in_feature, out_features, bias)
        # create a mask of ones for all weights (no element pruned at beginning)
        self.mask = Variable(torch.ones(self.weight.size()))

    def set_mask(self, mask=None):
        if mask is not None:
            self.mask = Variable(mask)
        self.weight.data = self.weight.data * self.mask.data

    def get_mask(self):
        return self.mask

    def get_weight_count(self):
        return self.mask.sum()

    def get_weight(self):
        return self.weight

    def forward(self, x):
        weight = self.weight.mul(self.mask)
        return F.linear(x, weight, self.bias)


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == MaskedLinearLayer:
            yield child


def get_weight_distribution(network):
    all_weights = []
    for layer in get_single_pruning_layer(network):
        mask = list(network.get_mask().abs().numpy().flatten())
        weights = list(network.get_weight().data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))

        all_weights += list(filtered_weights)
    return pd.DataFrame(data=all_weights)

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Standard feed-forward neural network with one hidden layer.
    Currently activation function: ReLU
    Output function: Log-Softmax
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = PruningLayer(nn.Linear(input_size, hidden_size))
        self.relu = nn.ReLU()
        self.fc2 = PruningLayer(nn.Linear(hidden_size, num_classes))
        self.log_softmax = F.log_softmax

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.log_softmax(out, dim=1)


class PruningLayer(nn.Module):
    """
    Pruning Layer is a Decorator for all nn.Modules with a named parameter `weight`.
    The parameter 'weight' will be changed by calling the prune method.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.mask = torch.ones(self.wrapped.weight.size())

    def get_mask(self):
        return self.mask

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        self.wrapped.weight.data = self.wrapped.weight.data * self.mask
        return self.wrapped.forward(x)

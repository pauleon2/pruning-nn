from pruninig_nn.network import PruningLayer
import random


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
            self.prune = strategy
        else:
            self.prune = random_pruning

    def prune(self, network, percentage):
        """
        :param: network: The network that should be pruned.
        :param: percentage: The percentage of weights that should be pruned.
        :return: The pruned network
        """
        self.prune(network, percentage)


def random_pruning(network, percentage):
    for child in get_single_pruning_layer(network):
        mask = child.get_mask()
        total = mask.sum()  # All parameters that are non zero can be pruned
        prune_goal = percentage * total
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


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == PruningLayer:
            yield child

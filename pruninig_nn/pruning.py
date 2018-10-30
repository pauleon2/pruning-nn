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

    def prune(self, network, percentage=0.1):
        """
        :param: network: The network that should be pruned.
        :param: percentage: The percentage of weights that should be pruned.
        :return: The pruned network
        """
        self.prune(network, percentage)


def random_pruning(network, percentage):
    pass

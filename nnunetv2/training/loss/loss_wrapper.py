from torch import nn
from typing import List, Union


__author__ = ["Riccardo Biondi"]
__email__ = ["riccardo.biondi7@unibo.it"]


class LossWrapper(nn.Module):
    """
    Utility class allowing to combine a list of losses all toghether.
    The combination is made in a weighted mode, please notice that the weights do non 
    necessary have to sum to 1.
    There is no weight scheduler, therefore the weight is constant across the whole training.
    """

    def __init__(self, losses: list, weights: list):
        """
        LossWrapper initializer. It will take the responsability to check if the 
        losses are actual loss (i.e. subclasses of nn.Module), the weights are actual weights (i.e. float or int)
        ant the losses and weights have the same len.

        Parameters
        ----------
        losses: List[nn.Module]
            list of losses to combine. The loss must be class instances.
        weights: List[Union[int, float]]
            a list of int and/or floats to use to weight the different losses.
        """
        
        super(LossWrapper, self).__init__()

        assert len(losses) == len(weights)

        # TODO: add a sanity check on the weights and loss types.
        self.losses = losses
        self.weights = weights

    def forward(self, x, y, loss_mask=None):
        
        # here gthe combination of the results. 
        # firstly, compute the list of the results, then weight them and sum them
        return sum( [weight * loss(x, y, loss_mask) for weight, loss in zip(self.weights, self.losses)])


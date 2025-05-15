import nnunetv2
import torch
import torch.nn as nn
from typing import Callable, Optional
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import load_json, join


__author__ = ["Riccardo Biondi"]
__email__ = ["riccardo.biondi7@unibo.it"]



class SobelGradientMagnitude(nn.Module):
    """
    Computes the gradient magnitude of an input tensor using the Sobel operator.

    This module supports both 2D and 3D data. It applies Sobel filters along spatial
    directions (x, y, and optionally z) and returns the L2-norm of the gradient vector
    at each spatial location.

    The gradient magnitude is computed as:

    .. math::

        G = \sqrt{G_x^2 + G_y^2 + (G_z^2) + \epsilon}

    where :math:`G_x`, :math:`G_y`, and (optionally) :math:`G_z` are the directional
    gradients, and :math:`\epsilon` is a small smoothing constant to ensure numerical stability.

    :param dimension: Dimensionality of the input data (2 for 2D, 3 for 3D).
    :type dimension: int
    :param smooth: A small value added to the squared gradient sum to avoid division by zero.
    :type smooth: float

    :raises ValueError: If `dimension` is not 2 or 3.
    """

    conv: Callable[..., torch.Tensor]

    def __init__(self, dimension: int = 2, smooth: float = 1e-7):
        """
        Initializes the SobelGradientMagnitude module.

        :param dimension: Spatial dimensionality of the input tensor. Must be 2 (for 2D) or 3 (for 3D).
        :type dimension: int
        :param smooth: Small constant added to the gradient magnitude for numerical stability.
        :type smooth: float

        :raises ValueError: If `dimension` is not 2 or 3.
        """
        super().__init__()

        self.smooth = smooth
        if (dimension != 2) & (dimension != 3):
            raise ValueError(f"Expected image or volume dimension be 2 or 3, {dimension} received instead")

        self.dimension = dimension

        # Set the appropriate convolution operation
        if dimension == 2:
            self.conv = torch.nn.functional.conv2d
            # Register 2D Sobel kernels for x and y directions
            self.register_buffer("kernelx", self._get_kernel(dimension, "x"))
            self.register_buffer("kernely", self._get_kernel(dimension, "y"))
        else:
            self.conv = torch.nn.functional.conv3d
            # Register 3D Sobel kernels for x, y, and z directions
            self.register_buffer("kernelx", self._get_kernel(dimension, "x"))
            self.register_buffer("kernely", self._get_kernel(dimension, "y"))
            self.register_buffer("kernelz", self._get_kernel(dimension, "z"))

    def _get_kernel(self, dimension: int, direction: str) -> torch.Tensor:
        """
        Returns the Sobel kernel tensor for the specified dimension and direction.

        :param dimension: The spatial dimensionality (2 or 3).
        :type dimension: int
        :param direction: The direction of the gradient. One of "x", "y", or "z".
        :type direction: str

        :return: A tensor representing the Sobel kernel for the specified direction and dimension.
        :rtype: torch.Tensor

        :raises ValueError: If an invalid direction is provided for the given dimension.
        """
        if (dimension == 2) & (direction == "x"):
            return torch.tensor([[[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]]])
        elif (dimension == 2) & (direction == "y"):
            return torch.tensor([[[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]]])
        elif (dimension == 3) & (direction == "x"):
            return torch.tensor([[[[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]],
                                   [[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                   [[-1, -2, -1],
                                    [-2, -4, -2],
                                    [-1, -2, -1]]]])
        elif (dimension == 3) & (direction == "y"):
            return torch.tensor([[[[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]],
                                   [[2, 4, 2],
                                    [0, 0, 0],
                                    [-2, -4, -2]],
                                   [[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]]]])
        elif (dimension == 3) & (direction == "z"):
            return torch.tensor([[[[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]],
                                   [[2, 0, -2],
                                    [4, 0, -4],
                                    [2, 0, -2]],
                                   [[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]]]])
        else:
            raise ValueError(f"Invalid direction '{direction}' for dimension {dimension}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sobel filtering to compute the gradient magnitude of the input tensor.

        :param x: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D data.
                The convolution is applied independently per channel.
        :type x: torch.Tensor

        :return: A tensor of the same shape as `x`, containing gradient magnitudes.
        :rtype: torch.Tensor

        :note: For 2D data, only x and y directional gradients are computed.
            For 3D data, x, y, and z gradients are used.
        """
        ndims = (x.ndim -1) * [1]
        to_repeat = [x.shape[0], *ndims]
        # Compute gradient in x direction
        g_x = self.conv(x, weight=torch.as_tensor(self.kernelx.repeat(to_repeat), device=x.device, dtype=torch.float),
                        padding="same", groups=x.shape[1])
        # Compute gradient in y direction
        g_y = self.conv(x, weight=torch.as_tensor(self.kernely.repeat(to_repeat), device=x.device, dtype=torch.float),
                        padding="same", groups=x.shape[1])

        # For 3D data, also compute z-direction gradient
        if self.dimension == 3:
            g_z = self.conv(x, weight=torch.as_tensor(self.kernelz.repeat(to_repeat), device=x.device, dtype=torch.float),
                            padding="same", groups=x.shape[1])
            
            return torch.sqrt(torch.pow(g_x, 2.) + torch.pow(g_y, 2.) + torch.pow(g_z, 2.) + self.smooth)
        # For 2D data, only x and y gradients are used
        return torch.sqrt(torch.pow(g_x, 2.) + torch.pow(g_y, 2.) + self.smooth)




class GradientLoss(nn.Module):
    """
    Computes a gradient-based loss for segmentation tasks, designed for use in the nnU-Net framework.

    This loss penalizes differences in the spatial gradients of the predicted segmentation map
    and the target segmentation map. It is useful for encouraging sharper boundaries and 
    better alignment of edges between predictions and targets.

    The gradients are computed using a configurable gradient computation module (e.g., Sobel filter),
    and the loss is normalized by the gradient magnitude of the target to make it scale-invariant.

    :param apply_nonlin: Optional name of a non-linearity to apply to the network output before
                        computing the gradient (e.g., ``"softmax_helper_dim1"``). If ``None``, no 
                        non-linearity is applied. 
    :type apply_nonlin: Optional[str]
    :param norm: Norm to use when computing the gradient difference. Must be either ``"l1"`` or ``"l2"``.
    :type norm: str
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
                    If ``None``, no reduction is applied.
    :type reduction: Optional[str]
    :param smooth: A small value added to the denominator to avoid division by zero when normalizing the loss.
    :type smooth: float
    :param gradient_computer: Name of the class to use for gradient computation (e.g., ``"SobelGradientMagnitude"``).
                            This class must be defined in ``nnunetv2.training.loss``.
    :type gradient_computer: str
    :param gradient_computer_kws: Keyword arguments passed to the gradient computation module.
    :type gradient_computer_kws: dict

    :raises ValueError: If an unsupported norm or reduction method is provided.

    :example:

        >>> criterion = GradientLoss(apply_nonlin="softmax_helpe_dim1", norm="l1", reduction="mean")
        >>> loss = criterion(predictions, ground_truth)

    :note: If the shapes of ``x`` and ``y`` differ, the ground truth ``y`` is assumed to be a label map
            and will be converted to a one-hot representation.

    :note: The gradient computation module must have a compatible interface, returning gradient
            magnitudes with the same shape as the input.
    """

    def __init__(
        self,
        apply_nonlin: Optional[str] = None,
        norm: str = "l1",
        reduction: Optional[str] = None,
        smooth: float = 1e-7, 
        gradient_computer: str = "SobelGradientMagnitude",
        gradient_computer_kws: dict = {}
    ):
        """
        Initializes the GradientLoss.

        Args:
            apply_nonlin (Callable, optional): Non-linearity to apply to predictions (e.g., softmax).
            norm (str): Norm to compute gradient differences ('l1' or 'l2').
            reduction (str or None): Reduction method to apply to the loss ('mean', 'sum', or None).
            gradient_computer (str): Name of the gradient computation module (currently supports only 'SobelGradientMagnitude').
            gradient_computer_kws (dict): Additional keyword arguments for gradient computation module.

        Raises:
            ValueError: If `norm` is not 'l1' or 'l2'.
            ValueError: If `reduction` is not one of [None, 'sum', 'mean'].
        """
        super(GradientLoss, self).__init__()

        self.smooth = smooth

        if norm not in ["l1", "l2"]:
            raise ValueError(f"Norm {norm} is not supported. Please use one of ['l1', 'l2'].")

        if reduction not in ["none", "sum", "mean"]:
            raise ValueError(f"{reduction} is not supported. Please use one of ['none', 'sum', 'mean'].")

        # Gradient computation module (Sobel for now)

        self.gradient = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "loss"),
                gradient_computer,
                current_module="nnunetv2.training.loss")(**gradient_computer_kws)
        

        self.apply_nonlin = apply_nonlin

        if apply_nonlin is not None:
            self.apply_nonlin = recursive_find_python_class(join(nnunetv2.__path__[0], "utilities"), apply_nonlin, current_module="nnunetv2.utilities")
        # Select the norm function
        self.norm = torch.nn.functional.mse_loss if norm == "l2" else torch.nn.functional.l1_loss
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None) -> torch.Tensor:
        """
        Computes the gradient-based loss between the network predictions and ground truth.

        The method calculates the normalized difference between the spatial gradient magnitudes
        of the prediction and the ground truth. An optional non-linearity (e.g., softmax) can
        be applied to the prediction before computing the gradients. If the ground truth is 
        not already one-hot encoded, it is converted internally.

        :param x: The raw network output tensor of shape (B, C, H, W) or (B, C, D, H, W),
                where B is the batch size and C is the number of classes.
        :type x: torch.Tensor
        :param y: The ground truth tensor. It can be of shape (B, C, H, W) or (B, C, D, H, W) 
                if already one-hot encoded, or of shape (B, 1, H, W) / (B, 1, D, H, W) if it 
                contains class indices.
        :type y: torch.Tensor
        :param loss_mask: Optional mask to apply to the computed loss. (Currently unused.)
        :type loss_mask: Optional[torch.Tensor]

        :return: A scalar tensor representing the gradient loss. The type depends on the reduction
                mode: if ``'none'``, it may return a tensor of per-element losses.
        :rtype: torch.Tensor

        :note: The loss is computed as:

            .. math::

                L = \\frac{\\text{Norm}(\\nabla \\hat{y}, \\nabla y)}{\\text{Norm}(\\nabla y, 0) + \\epsilon}

            where :math:`\\hat{y}` is the prediction, :math:`y` is the ground truth, 
            and Norm refers to either L1 or L2 norm over the gradients.

        :raises: No explicit exceptions are raised within this method, but invalid input shapes
                or types may cause runtime errors during tensor operations.
        """

        if x.shape == y.shape:
        #   # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            with torch.no_grad():
                y_onehot = torch.cat([(y.long() == i).long() for i in range(0, x.shape[1])], dim=1)
                y_onehot = y_onehot.float()

        # Apply optional non-linearity (e.g., softmax) to predictions
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Compute gradient magnitudes
        x_grad = self.gradient(x)
        y_grad = self.gradient(y_onehot)

        # Compute difference between gradients using the selected norm
        # this will be the numerator of the final loss
        num = self.norm(x_grad, y_grad, reduction=self.reduction)
        den = self.norm(y_grad, torch.as_tensor(torch.zeros(y_grad.shape), device=y_grad.device), reduction=self.reduction) + self.smooth
        loss = num / den

        return loss

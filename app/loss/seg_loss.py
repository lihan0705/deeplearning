import torch.nn as nn
import torch


class SegCrossEntropyLoss(object):
    def __init__(self, ignore_nan_targets=False, num_classes=3) -> None:
        self.ignore_nan_targets = ignore_nan_targets
        self.num_classes = num_classes

    def _compute_loss(self, prediction_tensor, target_tensor):
        """Method to be overridden by implementations.

        Args:
            prediction_tensor: a tensor representing predicted quantities
            target_tensor: a tensor representing regression or classification targets
            **params: Additional keyword arguments for specific implementations of
                    the Loss.

        Returns:
            loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """
        ignore_label = 0
        if self.ignore_nan_targets:
            target_tensor[target_tensor >= self.num_classes] = ignore_label
            target_tensor[target_tensor <= 0] = ignore_label
        loss = nn.CrossEntropyLoss()
        return loss(prediction_tensor, target_tensor)

import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class GCNMBHead(BaseHead):
    """The classification head for Slowfast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initializ the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_components=['j', 'b'],
                 loss_weights=[1., 1.],
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        if isinstance(dropout, float):
            dropout = {'j': dropout, 'b': dropout}
        assert isinstance(dropout, dict)

        self.dropout = dropout
        self.init_std = init_std
        self.in_channels = in_channels

        self.loss_components = loss_components
        if isinstance(loss_weights, float):
            loss_weights = [loss_weights] * len(loss_components)

        assert len(loss_weights) == len(loss_components)
        self.loss_weights = loss_weights
        if self.dropout['j'] != 0:
            self.dropout_j = nn.Dropout(p=self.dropout['j'])
            self.dropout_b = nn.Dropout(p=self.dropout['b'])
        else:
            self.dropout = None


        self.fc_j = nn.Linear(in_channels[0], num_classes)
        self.fc_b = nn.Linear(in_channels[1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_j, std=self.init_std)
        normal_init(self.fc_b, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x_j = x[0]
        x_b = x[1]
        x_j, x_b = self.avg_pool(x_j), self.avg_pool(x_b)

        x_j = x_j.view(x_j.size(0), -1)
        x_b = x_b.view(x_b.size(0), -1)
        if self.dropout is not None:
            x_j = self.dropout_j(x_j)
            x_b = self.dropout_b(x_b)
        # print(x_b.shape)
        # exit(0)
        cls_scores = {}
        cls_scores['j'] = self.fc_j(x_j)
        cls_scores['b'] = self.fc_b(x_b)

        return cls_scores

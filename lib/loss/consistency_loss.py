import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):

    def __init__(self, n_neg, margin, eps=1e-8):
        """
        computes L_y as in paper
        :param n_neg: number of negative pairs per positive pair
        :param margin: margin as in max(0, margin + s_neg - s_pos)
        """
        super(ConsistencyLoss, self).__init__()
        self.n_neg = n_neg
        self.margin = margin
        self.eps = eps
        raise NotImplementedError

    def forward(self, x, y):
        """
        assumes that each x corresponds with n_samp ys
        """

        raise NotImplementedError
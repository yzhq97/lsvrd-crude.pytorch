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

    def forward(self, s, p):
        """
        N = batch_size
        :param s: [N, N] pairwise similarity matrix
        s[i, j] = s(xi, xj)
        :param p: [N, N] positivity matrix
        p[i, j] = 1 means that (xi, xj) is a positive pair
        """

        raise NotImplementedError
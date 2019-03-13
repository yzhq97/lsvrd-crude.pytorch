import torch
import torch.nn as nn

class TripletLoss(nn.Module):

    def __init__(self, n_neg, margin):
        """
        computes L_y as in paper
        :param n_neg: number of negative samples per positive sample
        """
        super(TripletLoss, self).__init__()
        assert n_neg > 0

        self.n_neg = n_neg
        self.margin = margin

    def forward(self, s):
        """
        :param s: [N, N] pairwise similarity matrix
        s[i, j] = s(xi, yj)
        """
        N1, N2 = s.size()
        assert N1 == N2

        s_negs = s[1-torch.eye(N1)]

        return loss






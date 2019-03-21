import torch
import torch.nn as nn

class TripletLoss(nn.Module):

    def __init__(self, n_neg, margin, eps=1e-8):
        """
        computes L_y as in paper
        :param n_neg: number of negative pairs per positive pair
        :param margin: margin as in max(0, margin + s_neg - s_pos)
        """
        super(TripletLoss, self).__init__()
        self.n_neg = n_neg
        self.margin = margin
        self.eps = eps

    def forward(self, s):
        """
        N = batch_size
        assume that only diagonal elements are positive pairs

        :param s: [N, N] pairwise similarity matrix
        s[i, j] = s(xi, yj)
        """

        N1, N2 = s.size()
        assert N1 == N2
        assert self.n_neg <= N1 - 1
        N = N1

        p = torch.eye(N, dtype=torch.int8)

        pos_ind = p.nonzero()
        pos_s = s[pos_ind[:, 0], pos_ind[:, 1]].view([N, 1])

        neg_ind = p.neg().add(1).nonzero()
        neg_s = s[neg_ind[:, 0], neg_ind[:, 1]].view([N, N-1])

        loss = neg_s - pos_s + self.margin # [ N, N-1 ]
        eps = torch.ones_like(loss).mul(self.eps)
        loss = torch.where(loss > 0, loss, eps)

        loss, _ = loss.topk(k=self.n_neg, dim=1, sorted=False)
        loss = loss.mean()

        return loss





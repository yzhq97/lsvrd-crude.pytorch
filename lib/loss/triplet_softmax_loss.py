import torch
import torch.nn as nn

class TripletSoftmaxLoss(nn.Module):

    def __init__(self, n_neg, margin, eps=1e-8):
        """
        computes L_y as in paper
        :param n_neg: number of negative pairs per positive pair
        :param margin: margin as in max(0, margin + s_neg - s_pos)
        """
        super(TripletSoftmaxLoss, self).__init__()
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
        s = torch.exp(s)

        pos_ind = p.nonzero()
        pos_s = s[pos_ind[:, 0], pos_ind[:, 1]] # [N]

        neg_ind = p.neg().add(1).nonzero()
        neg_s = s[neg_ind[:, 0], neg_ind[:, 1]].view([N, N-1])

        neg_s, _ = neg_s.topk(k=self.n_neg, dim=1, sorted=False)
        neg_s = neg_s.sum(dim=1) # [ N ]
        loss = torch.log(pos_s / (pos_s + neg_s)).neg()
        loss = loss.mean()

        return loss




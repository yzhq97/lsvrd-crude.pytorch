import torch
import torch.nn as nn

class TripletLoss(nn.Module):

    def __init__(self, n_neg, margin):
        """
        :param n_neg: number of negative samples per positive sample
        """
        super(TripletLoss, self).__init__()
        self.n_neg = n_neg
        self.margin = margin

    def forward(self, s, px, pxy):
        """
        :param s: [B, Nx, Ny] pairwise similarity matrix

        :param px: [B, Nx]
        px(i)=1 means xi is a postive sample
        px(i)=0 means xi is a negative sample

        :param pxy: [B, Nx, Ny] positivity matrix.
        pxy(i, j)=1 means that (xi, yj) is a postive pair
        pxy(i, j)=0 means that (xi, yj) is a negative pair
        """

        B, Nx, Ny = s.size()

        pos_indices = torch.nonzero(px) # [ Nx_pos, 2 ]
        s_pos = s[pos_indices] # [ Nx_pos, Ny ]

        pxy_neg = 1 - pxy # mark negative pairs with ones
        pxy_x_pos = pxy_neg[pos_indices] # [ Nx_pos, Ny ]




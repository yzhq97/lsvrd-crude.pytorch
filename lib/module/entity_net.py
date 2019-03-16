import torch.nn as nn

class EntityNet (nn.Module):

    def __init__(self, in_dim, hid_dim):
        super(EntityNet, self).__init__()

        self.layers_1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        self.layers_2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
        )

    def forward(self, x):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        B, C, H, W = x.size()
        x = x.view(B, -1)
        intermediate = self.layers_1(x)
        x = self.layers_2(intermediate)
        return x, intermediate


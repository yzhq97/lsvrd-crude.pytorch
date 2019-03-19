import torch.nn as nn

class EntityNet (nn.Module):

    def __init__(self, in_dim, crop_size, emb_dim):
        super(EntityNet, self).__init__()
        assert crop_size == 7

        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, int(emb_dim / 2), kernel_size=3, stride=2),
            nn.BatchNorm2d(int(emb_dim / 2)),
            nn.ReLU(),
            nn.Conv2d(int(emb_dim / 2), emb_dim, 3),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        B, C, H, W = x.size()
        intermediate = self.block1(x)
        intermediate = intermediate.squeeze()
        x = self.block2(intermediate)
        return x, intermediate


import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNet (nn.Module):

    def __init__(self, in_dim, crop_size, emb_dim):
        super(RelationNet, self).__init__()
        assert crop_size == 7

        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, int(emb_dim/2), kernel_size=3, stride=2),
            nn.BatchNorm2d(int(emb_dim/2)),
            nn.ReLU(),
            nn.Conv2d(int(emb_dim/2), emb_dim, 3),
            nn.BatchNorm2d(emb_dim)
        )

        self.l2 = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x, sbj_emb, sbj_inter, obj_emb, obj_inter):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        B, C, H, W = x.size()
        intermediate = self.l1(x)
        intermediate = intermediate.squeeze()
        intermediate_relu = F.relu(intermediate)
        x = torch.cat([sbj_inter, intermediate_relu, obj_inter], dim=1)
        x = self.l2(x)
        x = torch.cat([sbj_emb, x, obj_emb], dim=1)
        x = intermediate + self.l3(x)

        return x


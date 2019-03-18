import torch
import torch.nn as nn

class RelationNet (nn.Module):

    def __init__(self, in_dim, crop_size, emb_dim):
        super(RelationNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(crop_size * crop_size * in_dim, emb_dim),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        self.block3 = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

    def forward(self, x, sbj_emb, sbj_inter, obj_emb, obj_inter):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        B, C, H, W = x.size()
        x = x.view(B, -1)
        x = self.block1(x)
        x = torch.cat([sbj_inter, x, obj_inter], dim=1)
        x = self.block2(x)
        x = torch.cat([sbj_emb, x, obj_emb], dim=1)
        x = self.block3(x)

        return x


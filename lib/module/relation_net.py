import torch
import torch.nn as nn

class RelationNet (nn.Module):

    def __init__(self, in_dim, hid_dim):
        super(RelationNet, self).__init__()

        self.layers_1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        self.layers_2 = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
        )

        self.layers_3 = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

    def forward(self, x, sbj_emb, sbj_inter, obj_emb, obj_inter):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        B, C, H, W = x.size()
        x.view_(B, -1)
        x = self.layers_1(x)
        x = torch.cat([sbj_inter, x, obj_inter], dim=1)
        x = self.layers_2(x)
        x = torch.cat([sbj_emb, x, obj_emb], dim=1)
        x = self.layers_3(x)

        return x


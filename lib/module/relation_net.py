import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNet (nn.Module):

    def __init__(self, in_dim, crop_size, emb_dim):
        super(RelationNet, self).__init__()
        assert crop_size >= 3

        l1_layers = [
            nn.Conv2d(in_dim, emb_dim, kernel_size=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU()
        ]
        current_size = crop_size
        while current_size > 1:
            stride = 2 if current_size >= 5 else 1
            layer = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=stride)
            l1_layers.append(layer)
            l1_layers.append(nn.BatchNorm2d(emb_dim))
            l1_layers.append(nn.ReLU())
            current_size = current_size - 2 - (stride - 1) * 2
        self.l1 = nn.Sequential(*l1_layers)

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


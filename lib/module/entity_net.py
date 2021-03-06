import torch.nn as nn
import torch.nn.functional as F

class EntityNet (nn.Module):

    def __init__(self, in_dim, crop_size, emb_dim, hid_dim):
        super(EntityNet, self).__init__()
        assert crop_size >= 3 and crop_size % 2 == 1

        l1_layers = [
            nn.Conv2d(in_dim, hid_dim, kernel_size=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU()
        ]
        current_size = crop_size

        while current_size > 3:
            l1_layers.extend([
                nn.Conv2d(hid_dim, hid_dim, kernel_size=3),
                nn.BatchNorm2d(hid_dim),
                nn.ReLU()
            ])
            current_size = current_size - 2
        
        l1_layers.extend([
            nn.Conv2d(hid_dim, emb_dim, kernel_size=3),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU()
        ])
        self.l1 = nn.Sequential(*l1_layers)

        self.l2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        """
        :param x: [ B, C, aligned_h, aligned_w ]
        """
        intermediate = self.l1(x)
        B, C, _, _ = intermediate.size()
        intermediate = intermediate.view(B, C)
        intermediate_relu = F.relu(intermediate)
        x = intermediate + self.l2(intermediate_relu)
        return x, intermediate_relu


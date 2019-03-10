import torch
import torch.nn as nn

class LSVRD(nn.Module):

    def __init__(self, lm,
                 sbj_net, obj_net, rel_net,
                 sbj_loss, obj_loss, rel_loss):

        super(LSVRD, self).__init__()

    def forward(self, fm, boxes):
        pass
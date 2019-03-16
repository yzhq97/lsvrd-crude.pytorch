import torch.nn as nn
import torchvision.models

class FeatureNet (nn.Module):

    def __init__(self):

        super(FeatureNet, self).__init__()

        cnn = torchvision.models.resnet101(pretrained=True)
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]

        for i in range(1, 5):
            name = 'layer%d' % (i)
            layers.append(getattr(cnn, name))

        self.cnn = nn.Sequential(*layers)


    def forward(self, x):
        """
        :param x: [ B, C, H, W ]
        """
        x = self.cnn(x)
        return x


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
            cnn.layer1,
            cnn.layer2,
            cnn.layer3,
            cnn.layer4
        ]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: [ B, C, H, W ]
        """
        x = self.cnn(x)
        return x

    def freeze(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def defreeze(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
import torch.nn as nn
import torchvision.models

class FeatureNet (nn.Module):

    def __init__(self):

        super(FeatureNet, self).__init__()

        cnn = torchvision.models.resnet101(pretrained=True)
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1,
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        # self.avgpool = cnn.avgpool

    def forward(self, x):
        """
        :param x: [ B, C, H, W ]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        return x

    def freeze(self, layer="all"):
        if layer == "all":
            for param in self.parameters():
                param.requires_grad = False
        else:
            layer = getattr(self, layer)
            for param in layer.parameters():
                param.requires_grad = False

    def defreeze(self, layer="all"):
        if layer == "all":
            for param in self.parameters():
                param.requires_grad = True
        else:
            layer = getattr(self, layer)
            for param in layer.parameters():
                param.requires_grad = True
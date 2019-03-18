import torch.nn as nn
import torchvision.models

class ResNet (nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

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

class VGGNet (nn.Module):

    def __init__(self):

        super(VGGNet, self).__init__()

        cnn = torchvision.models.vgg19(pretrained=True)
        self.features = cnn.features

        self.layers_dict = {}
        block_idx = 1
        conv_dix = 1
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                self.layers_dict["conv%d_%d" % (block_idx, conv_dix)] = layer
                conv_dix += 1
            elif isinstance(layer, nn.ReLU):
                self.layers_dict["relu%d_%d" % (block_idx, conv_dix)] = layer
            elif isinstance(layer, nn.MaxPool2d):
                self.layers_dict["pool%d" % block_idx] = layer
                block_idx += 1
                conv_dix = 1

    def forward(self, x):
        """
        :param x: [ B, C, H, W ]
        """
        x = self.features(x)

        return x

    def freeze(self, layer="all"):
        if layer == "all":
            for param in self.parameters():
                param.requires_grad = False
        else:
            layer = self.layers_dict[layer]
            for param in layer.parameters():
                param.requires_grad = False

    def defreeze(self, layer="all"):
        if layer == "all":
            for param in self.parameters():
                param.requires_grad = True
        else:
            layer = self.layers_dict[layer]
            for param in layer.parameters():
                param.requires_grad = True

backbones = {
    "resnet101": ResNet,
    "vgg19": VGGNet
}
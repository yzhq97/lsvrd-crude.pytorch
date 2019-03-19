import torch.nn as nn
import torchvision.models

class ResNet (nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        cnn = torchvision.models.resnet101(pretrained=True)
        self.cnn = cnn
        self.features = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool,
                                      cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4)
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
        x = self.features(x)

        return x

    def freeze(self, layer="all"):
        if layer == "all":
            self.train(False)
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            layer = getattr(self, layer)
            layer.train(False)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def defreeze(self, layer="all"):
        if layer == "all":
            self.train(True)
            for param in self.parameters():
                param.requires_grad = True
        else:
            layer = getattr(self, layer)
            layer.train(True)
            for param in layer.parameters():
                param.requires_grad = True

class VGGNet (nn.Module):

    def __init__(self):

        super(VGGNet, self).__init__()

        self.cnn = torchvision.models.vgg19(pretrained=True)
        self.features = self.cnn.features

        self.layers_dict = {}
        block_idx = 1
        conv_dix = 1
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                self.layers_dict["conv%d_%d" % (block_idx, conv_dix)] = i
                conv_dix += 1
            elif isinstance(layer, nn.ReLU):
                self.layers_dict["relu%d_%d" % (block_idx, conv_dix)] = i
            elif isinstance(layer, nn.MaxPool2d):
                self.layers_dict["pool%d" % block_idx] = i
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
            self.train(False)
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            layer = self.layers_dict[layer]
            self.features[layer].train(False)
            self.features[layer].eval()
            for param in self.features[layer].parameters():
                param.requires_grad = False

    def defreeze(self, layer="all"):
        if layer == "all":
            self.train(True)
            for param in self.parameters():
                param.requires_grad = True
        else:
            layer = self.layers_dict[layer]
            self.features[layer].train(True)
            for param in self.features[layer].parameters():
                param.requires_grad = True

backbones = {
    "resnet101": ResNet,
    "vgg19": VGGNet
}
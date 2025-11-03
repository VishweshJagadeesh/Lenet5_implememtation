'''
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4M
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                """
                for the Imagenet dataset paper uses a conv layer to manage the dimensions
                """
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, imagenet=True):
        super(ResNet, self).__init__()
        if imagenet:
            option='B'
            self.in_planes = 64
            # ImageNet initial conv: 7x7 kernel, stride 2, padding 3
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            # ImageNet max pool
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # ImageNet layers use 64, 128, 256, 512 channels
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,option=option)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,option=option)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,option=option)
            # Added a fourth layer for ResNet-18 (which has 4 main stages)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,option=option) 
            final_in_features = 512 * block.expansion
        else:
            option='A'     
            self.in_planes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,option=option)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,option=option)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,option=option)
            final_in_features = 64 * block.expansion

        self.linear = nn.Linear(final_in_features, num_classes)
        self.imagenet = imagenet

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option='A'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,option=option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.imagenet:
            out = self.maxpool(out) # Add max-pooling for ImageNet
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out) # Add the fourth layer
            
        else: # CIFAR forward pass
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3],num_classes=10,imagenet=False)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5],num_classes=10,imagenet=False)
def resnet44():
    return ResNet(BasicBlock, [7, 7, 7],num_classes=10,imagenet=False)
def resnet56():
    return ResNet(BasicBlock, [9, 9, 9],num_classes=10,imagenet=False)
def resnet110():
    return ResNet(BasicBlock, [18, 18, 18],num_classes=10,imagenet=False)
def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200],num_classes=10,imagenet=False)

def resnet18():
    """
    Standard ResNet-18 for ImageNet (4 main layers, 2 BasicBlocks per layer)
    Default to num_classes=1000 and imagenet=True
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, imagenet=True)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
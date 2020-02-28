import torch.nn as nn
import torchvision.transforms as transforms
import math

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, Conv2d, stride=1):
    "3x3 convolution with padding"
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model, BatchNorm2d):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, relu, Conv2d, BatchNorm2d, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, Conv2d, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = relu()
        self.conv2 = conv3x3(planes, planes, Conv2d)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, relu, Conv2d, BatchNorm2d, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = relu()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, Conv2d, relu, BatchNorm2d, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, relu, Conv2d, BatchNorm2d, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu, Conv2d, BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        x = self.layer3(out_2)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, Conv2d=nn.Conv2d, Linear=nn.Linear, RELU=nn.ReLU, BatchNorm2d=nn.BatchNorm2d, num_classes=1000, block=Bottleneck, layers=[2, 2, 2, 2]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Conv2d, RELU, BatchNorm2d, block, 64, layers[0])
        self.layer2 = self._make_layer(Conv2d, RELU, BatchNorm2d, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Conv2d, RELU, BatchNorm2d, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Conv2d, RELU, BatchNorm2d, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = Linear(512 * block.expansion, num_classes)

        init_model(self, BatchNorm2d)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar(ResNet):

    def __init__(self, num_classes=10, depth=20, block=BasicBlock, width=1, Conv2d=nn.Conv2d, Linear=nn.Linear, relu=nn.ReLU, BatchNorm2d=nn.BatchNorm2d):
        super(ResNet_cifar, self).__init__()
        self.inplanes = 16*width
        n = int((depth - 2) / 6)
        self.conv1 = Conv2d(3, 16*width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16*width)
        self.relu = relu()
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(Conv2d, relu, BatchNorm2d, block, 16*width, n)
        self.layer2 = self._make_layer(Conv2d, relu, BatchNorm2d, block, 32*width, n, stride=2)
        self.layer3 = self._make_layer(Conv2d, relu, BatchNorm2d, block, 64*width, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        if num_classes == 200:
            self.avgpool = nn.AvgPool2d(16)
        self.fc = Linear(64*width, num_classes)

        init_model(self, BatchNorm2d)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            100: {'lr': 1e-2},
            150: {'lr': 1e-3},
        }


def resnet(**kwargs):
    num_classes, depth, dataset, ReLU = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'ReLU'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 20 #56
        if ReLU == 'ReLU':
            nonlinearity = nn.ReLU
        elif ReLU == 'ELU':
            nonlinearity = nn.ELU
        else:
            nonlinearity = nn.Softplus
        print('nonlinearity',nonlinearity)
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth, nonlinearity=nonlinearity)


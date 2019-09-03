import os
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
from models.resnet import ResNet_cifar

def get_model(args):

    if 'cifar' in args.dataset or 'tiny' in args.dataset:

        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise ValueError

        if args.model == 'ResNet20':
            model = ResNet_cifar(num_classes=num_classes, depth=20)
        elif args.model == 'ResNet32':
            model = ResNet_cifar(num_classes=num_classes, depth=32)
        elif args.model == 'ResNet44':
            model = ResNet_cifar(num_classes=num_classes, depth=44)
        elif args.model == 'ResNet56':
            model = ResNet_cifar(num_classes=num_classes, depth=56)
        elif args.model == 'ResNet110':
            model = ResNet_cifar(num_classes=num_classes, depth=110)
        elif args.model == 'WideResNet40_4':
            model = WideResNet(40, num_classes, nonlinearity, widen_factor=4, dropRate=0.0)
        elif args.model == 'DenseNet40_40':
            model = DenseNet3(40, num_classes, nonlinearity, growth_rate=40, reduction=0.5, bottleneck=True, dropRate=0.0)
        else:
            print('unknown model')
            raise ValueError

    elif args.dataset == imagenet:
        if args.model == "ResNet18":
            model = models.resnet18()
        elif args.model == "ResNet18_pretrained":
            model = models.resnet18(pretrained=True)
        elif args.model == "ResNet34":
            model = models.resnet34()
        elif args.model == "ResNet34_pretrained":
            model = models.resnet34(pretrained=True)
        else:
            raise NotImplementedError
    else:
        print('unknown data set')
        raise NotImplementedError

    if args.load_model:
        state = torch.load(args.load_model)['model']
        new_state = OrderedDict()
        for k in state:
            # naming convention for data parallel
            if 'module' in k:
                v = state[k]
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = state[k]
        model.load_state_dict(new_state)
        print('Loaded model from {}'.format(args.load_model))

    # Number of model parameters
    args.nparams = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: {}'.format(args.nparams))

    if args.cuda:
        if args.parallel_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    return model


def load_best_model(model, filename):
    if os.path.exists(filename):
        best_model_state = torch.load(filename)['model']
        model.load_state_dict(best_model_state)
        print('Loaded best model from {}'.format(filename))
    else:
        print('Could not find best model')

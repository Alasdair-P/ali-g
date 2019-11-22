import os
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
from models.resnet import ResNet_cifar
from models.wide_resnet import WideResNet
from models.mlp import MLP


def get_model(args):

    if args.opt == 'cgd' or args.active_func == 'softplus':
        nonlinearity = nn.Softplus
    else:
        nonlinearity = nn.ReLU

    if 'spiral' in args.dataset:
        model = MLP(num_classes=args.n_classes, width=args.width, depth=args.depth)

    elif 'cifar' in args.dataset or 'tiny' in args.dataset:

        if args.model == 'resnet':
            model = ResNet_cifar(num_classes=args.n_classes, depth=args.depth, relu=nonlinearity)
        elif args.model == "dn":
            model = DenseNet3(args.depth, args.n_classes, args.growth, bottleneck=bool(args.bottleneck), dropRate=args.dropout)
        elif args.model == "wrn":
            model = WideResNet(args.depth, args.n_classes, args.width, dropRate=args.dropout)
        else:
            print('unknown model')
            raise ValueError

    elif args.dataset == 'imagenet':
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

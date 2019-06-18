import os
import torch
import torch.nn as nn
from .densenet import DenseNet3
from .wide_resnet import WideResNet
from collections import OrderedDict


def get_model(args):

    activation_func = nn.ReLU
    if args.activation_func == 'elu':
        activation_func = nn.ELU
    elif args.activation_func == 'softplus':
        activation_func = nn.Softplus
    else:
        print('the choice of activation function: ', args.activation_func, 'is not implemented please use relu, elu or softplus')
        raise ValueError

    if args.opt == 'cgd':
        activation_func = nn.Softplus

    if args.model == "dn":
        model = DenseNet3(args.depth, args.n_classes, activation_func, args.growth,
                          bottleneck=bool(args.bottleneck), dropRate=args.dropout)
    elif args.model == "wrn":
        model = WideResNet(args.depth, args.n_classes, activation_func, args.width, dropRate=args.dropout)
    else:
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

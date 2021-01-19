import os
import torch
import torchvision.models as th_models

from .densenet import DenseNet3
from .wide_resnet import WideResNet
from .mlp import MLP
from collections import OrderedDict
from .gnn import GNN
from ogb.graphproppred import PygGraphPropPredDataset


def get_model(args):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if 'mol' in args.dataset:
        dataset = PygGraphPropPredDataset(name = args.dataset)
    if args.model == "dn":
        model = DenseNet3(args.depth, args.n_classes, args.growth,
                          bottleneck=bool(args.bottleneck), dropRate=args.dropout)
    elif args.model == "wrn":
        model = WideResNet(args.depth, args.n_classes, args.width, dropRate=args.dropout)
    elif args.model == "mlp":
        model = MLP(args.depth, args.n_classes, args.width, args.input_dims)
    elif args.dataset == 'imagenet':
        model = th_models.__dict__[args.model](pretrained=False)
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    elif args.model == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = False)
    elif args.model == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = True)
    elif args.model == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = False)
    elif args.model == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = True)
    else:
        raise NotImplementedError
    """
    elif args.model == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = False).to(device)
    elif args.model == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = True).to(device)
    elif args.model == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = False).to(device)
    elif args.model == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = True).to(device)
    """

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

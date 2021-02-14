import os
import torch
import torchvision.models as th_models
import pandas as pd

from .densenet import DenseNet3
from .wide_resnet import WideResNet
from .mlp import MLP
from collections import OrderedDict
from .gnn_mol import GNN
from .gnn_code import GNN_CODE
from ogb.graphproppred import PygGraphPropPredDataset
from utils_gnn import ASTNodeEncoder, get_vocab_mapping


def get_model(args):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if 'mol' in args.dataset or 'code' in args.dataset:
        dataset = PygGraphPropPredDataset(name = args.dataset)
        virtual_node = True if 'virtual' in args.model else False
        gnn_type = 'gin' if 'gin' in args.model else 'gcn'
        root = '{}/{}'.format(os.environ['GRAPH_DATA'], args.dataset)

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
    elif 'mol' in args.dataset:
        model = GNN(gnn_type = gnn_type, num_tasks = dataset.num_tasks, num_layer = args.depth, emb_dim = args.width, drop_ratio = args.dropout, virtual_node = virtual_node)
    elif 'code' in args.dataset:
        nodetypes_mapping = pd.read_csv(os.path.join(root, 'mapping', 'typeidx2type.csv.gz'))
        nodeattributes_mapping = pd.read_csv(os.path.join(root, 'mapping', 'attridx2attr.csv.gz'))
        node_encoder = ASTNodeEncoder(args.width, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
        model = GNN_CODE(num_vocab = 5002, max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.depth, emb_dim = args.width,  gnn_type = gnn_type, virtual_node = virtual_node, drop_ratio = args.dropout)
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

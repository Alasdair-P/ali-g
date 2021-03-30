# top-import for cuda device initialization
from cuda import set_cuda
from cuda_jade import set_cuda_jade

import mlogger
import torch

from cli import parse_command
from loss import get_loss
from utils import setup_xp, set_seed, save_state, write_results
from data import get_data_loaders
from models import get_model, load_best_model
from optim import get_optimizer, decay_optimizer
from epoch import train, test, test_rank, train_graph, test_graph
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def main(args):
    print('beginning main')
    print('jade:',args.jade)
    if args.jade:
        set_cuda_jade(args)
    else:
        set_cuda(args)
    set_seed(args)

    loader_train, loader_val, loader_test = get_data_loaders(args)
    loss = get_loss(args)
    model = get_model(args)
    print([module for module in model.modules() if type(module) != torch.nn.Sequential][0])
    optimizer = get_optimizer(args, model, loss, parameters=model.parameters())
    save_state(model, optimizer, '{}/x_0.pkl'.format(args.xp_name))
    xp = setup_xp(args, model, optimizer)
    if 'ogbg' in args.dataset:
        evaluator = Evaluator(args.dataset)
        dataset = PygGraphPropPredDataset(name = args.dataset)
    for i in range(args.epochs):
        xp.epoch.update(i)
        if 'ogbg' in args.dataset:
            train_graph(model, loss, optimizer, evaluator, dataset, loader_train, args, xp)
            test_graph(model, optimizer, evaluator, dataset, loader_val, args, xp)
        else:
            train(model, loss, optimizer, loader_train, args, xp)
            test(model, optimizer, loader_val, args, xp)
        if args.opt ==  'alig2':
            optimizer.epoch()
        if (i + 1) in args.T:
            decay_optimizer(args, optimizer, args.decay_factor)
            if args.opt ==  'alig2':
                optimizer.update_lb()
                # load_best_model(model, '{}/x_0.pkl'.format(args.xp_name))

    load_best_model(model, '{}/best_model.pkl'.format(args.xp_name))
    if args.loss == 'map' or args.loss  == 'ndcg':
        test_rank(model, loss, optimizer, loader_test, args, xp)
    elif 'ogbg' in args.dataset:
        test_graph(model, optimizer, evaluator, dataset, loader_test, args, xp)
    else:
        test(model, optimizer, loader_test, args, xp)
    write_results(args, xp, '.')


if __name__ == '__main__':
    args = parse_command()
    with mlogger.stdout_to("{}/log.txt".format(args.xp_name), enabled=args.log):
        main(args)

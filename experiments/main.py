# top-import for cuda device initialization
from cuda import set_cuda

import mlogger

from cli import parse_command
from loss import get_loss
from utils import setup_xp, set_seed, save_state, write_results
from data import get_data_loaders
from models import get_model, load_best_model
from optim import get_optimizer, decay_optimizer
from epoch import train, test, test_rank


def main(args):
    print('begin main')
    set_cuda(args)
    set_seed(args)

    print('after seed and cuda')
    loader_train, loader_val, loader_test = get_data_loaders(args)
    print('after dataset')
    loss = get_loss(args)
    print('after loss')
    model = get_model(args)
    print('after model')
    optimizer = get_optimizer(args, model, loss, parameters=model.parameters())
    print('after opt')
    xp = setup_xp(args, model, optimizer)

    print('after xp')
    print('begin main loop')
    for i in range(args.epochs):
        xp.epoch.update(i)
        train(model, loss, optimizer, loader_train, args, xp)
        if args.loss == 'map' or args.loss  == 'ndcg':
            test_rank(model, loss, optimizer, loader_val, args, xp)
        else:
            test(model, optimizer, loader_val, args, xp)
        if (i + 1) in args.T:
            decay_optimizer(args, optimizer, args.decay_factor)

    load_best_model(model, '{}/best_model.pkl'.format(args.xp_name))
    if args.loss == 'map' or args.loss  == 'ndcg':
        test_rank(model, loss, optimizer, loader_test, args, xp)
    else:
        test(model, optimizer, loader_test, args, xp)
    write_results(args, xp, args.xp_name)
    write_results(args, xp, args.log_dir)
    write_results(args, xp, './')


if __name__ == '__main__':
    args = parse_command()
    with mlogger.stdout_to("{}/log.txt".format(args.xp_name), enabled=args.log):
        main(args)

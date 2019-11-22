# top-import for cuda device initialization
from cuda import set_cuda

import mlogger

from cli import parse_command
from loss import get_loss
from utils import setup_xp, set_seed, save_state
from data import get_data_loaders
from data.spiral import plot_decsion_boundary
from models.main import get_model, load_best_model
from optim import get_optimizer, decay_stuff
from epoch import train, test
from reg import Reg


def main(args):

    set_cuda(args)
    set_seed(args)

    loader_train, loader_val, loader_test = get_data_loaders(args)
    loss = get_loss(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model, loss, parameters=model.parameters())
    xp = setup_xp(args, model, optimizer)
    reg = Reg(args, model)

    i = 0
    for i in range(args.epochs):
        xp.epoch.update(i)
        reg.epoch_update()

        if i == args.hq_epoch and reg:
            reg.hard_quantize(optimizer)

        train(model, loss, optimizer, loader_train, args, xp, reg)
        test(model, optimizer, loader_val, args, xp, i)

        decay_stuff(xp, model, args, optimizer, loss, i)

    test(model, optimizer, loader_val, args, xp, i)
    test(model, optimizer, loader_test, args, xp, i)
    reg.calc_dist_to_binary()
    load_best_model(model, '{}/best_model.pkl'.format(args.xp_name))
    test(model, optimizer, loader_val, args, xp, i)
    test(model, optimizer, loader_test, args, xp, i)
    reg.calc_dist_to_binary()
    plot_decsion_boundary(model, args)


if __name__ == '__main__':
    args = parse_command()
    with mlogger.stdout_to("{}/log.txt".format(args.xp_name), enabled=args.log):
        main(args)

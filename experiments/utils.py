import os
import sys
import socket
import torch
import mlogger
import random
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('tensor board not found')


def regularization(model, l2):
    reg = 0.5 * l2 * sum([p.data.norm() ** 2 for p in model.parameters()]) if l2 else 0
    return reg


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        visdom_plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        visdom_plotter = None

    if args.tensorboard:
        print('args.tensorboard:', args.tensorboard)
        summary_writer = SummaryWriter(log_dir=args.tb_dir)
    else:
        summary_writer = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(visdom_plotter=visdom_plotter, summary_writer=summary_writer, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="loss")
    xp.train.lower_bound = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="lb")
    xp.train.kl = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="kl")
    xp.train.obj = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Weight-Norm")
    xp.train.grad_norm = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Grad-Norm")

    xp.train.alpha0 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Type", plot_legend="alpha0")
    xp.train.alpha1 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Type", plot_legend="alpha1")
    xp.train.alpha2 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Type", plot_legend="alpha2")
    xp.train.alpha3 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Type", plot_legend="alpha3")
    xp.train.alpha4 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Type", plot_legend="alpha4")

    xp.train.step_size = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='validation')

    xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')

    if args.loss == 'map':
        xp.max_val = mlogger.metric.Minimum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')
    else:
        xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter,  summary_writer=summary_writer, plot_title="Time", plot_legend='test')

    if args.dataset == "imagenet":
        xp.train.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="training")
        xp.val.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="validation")
        xp.test.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="test")

    if args.visdom:
        visdom_plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        visdom_plotter.set_win_opts("Objective", {'ytype': 'log'})
        # visdom_plotter.set_win_opts("Step-Type", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        if args.loss == 'map':
            xp.max_val.hook_on_new_min(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))
        else:
            xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

    return xp

def write_results(args, xp, path):
    if args.jade:
        file_name = 'jade_results.txt'
    else:
        file_name = 'results.txt'
    save_path = os.path.join(path, file_name)
    with open(save_path, 'a') as results:
        results.write('name,{xp_name},dataset,{ds},model,{model},opt,{opt},bs,{bs},eta,{eta},wd,{wd},max_norm,{mn},tr_acc,{tracc:.4f},val_acc,{vacc:.4f},te_acc,{teacc:.4f}\n'
                .format(ds=args.dataset,
                        model=args.model,
                        opt=args.opt,
                        bs=args.batch_size,
                        eta=args.eta,
                        wd=args.weight_decay,
                        mn=args.max_norm,
                        tracc=xp.train.acc.value,
                        vacc=xp.max_val.value,
                        teacc=xp.test.acc.value,
                        xp_name=args.xp_name))

@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)

    return 100. * acc

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

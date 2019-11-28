import torch

from tqdm import tqdm
# from dfw.losses import set_smoothing_enabled
from utils import accuracy, regularization

def train(model, loss, optimizer, loader, args, xp, reg):

    model.train()

    for metric in xp.train.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm, desc='Train Epoch',
                     leave=False, total=len(loader)):

        x, y = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        # forward pass
        optimizer.zero_grad()
        scores = model(x)

        # backward pass
        if args.opt == 'dfw':
            with set_smoothing_enabled(args.smooth_svm):
                loss_value, kl = loss(scores, y, x)
        else:
            loss_value, kl = loss(scores, y, x)

        if 'cgd' in args.opt or 'segd' in args.opt:
            if args.debug:
                loss_value.backward(create_graph=True)
            else:
                loss_value.backward()
            optimizer.step(lambda: float(loss_value), x, y)
        else:
            loss_value.backward()
            optimizer.step(lambda: float(loss_value))

        if reg:
            reg.iter_update()

        # monitoring
        batch_size = x.size(0)
        xp.train.acc.update(accuracy(scores, y), weighting=batch_size)
        xp.train.loss.update(float(loss_value), weighting=batch_size)
        xp.train.kl.update(float(kl), weighting=batch_size)
        xp.train.lower_bound.update(args.B, weighting=batch_size)
        xp.train.step_size.update(optimizer.step_size, weighting=batch_size)
        xp.train.step_size_u.update(optimizer.step_size_unclipped, weighting=batch_size)
        xp.train.step_type.update(optimizer.type_of_step, weighting=batch_size)

    xp.train.weight_norm.update(torch.sqrt(sum(p.norm() ** 2 for p in model.parameters())))
    xp.train.grad_norm.update(torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters())))
    xp.train.reg.update(0.5 * args.weight_decay * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.timer.update()



    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Loss {loss:.3f}\t'
          'Acc {acc:.2f}%\t'
          .format(int(xp.epoch.value),
                  timer=xp.train.timer.value,
                  acc=xp.train.acc.value,
                  obj=xp.train.obj.value,
                  loss=xp.train.loss.value))

    for metric in xp.train.metrics():
        metric.log(time=xp.epoch.value)


@torch.autograd.no_grad()
def test(model, optimizer, loader, args, xp, epoch):
    model.eval()

    if loader.tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm,
                     desc='{} Epoch'.format(loader.tag.title()),
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        scores = model(x)
        xp_group.acc.update(accuracy(scores, y), weighting=x.size(0))

    xp_group.timer.update()

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.2f}% \t'
          .format(int(xp.epoch.value),
                  tag=loader.tag.title(),
                  timer=xp_group.timer.value,
                  acc=xp_group.acc.value))

    if loader.tag == 'val':
        if args.hq_epoch and epoch >= args.hq_epoch:
            xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)
        elif args.hq_epoch:
            xp.max_val.update(0).log(time=xp.epoch.value)
        else:
            xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

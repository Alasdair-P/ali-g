import torch
import numpy as np
import time
from tqdm import tqdm
# from dfw.losses import set_smoothing_enabled
from utils import accuracy, regularization
from ogb.graphproppred import PygGraphPropPredDataset


def train_graph(model, loss, optimizer, evaluator, dataset, loader, args, xp):
    model.train()
    y_true = []
    y_pred = []

    for metric in xp.train.metrics():
        metric.reset()

    for step, batch_ in enumerate(tqdm(loader, desc="Iteration")):
        idx, batch = batch_
        # batch = batch.cuda() if args.cuda else batch
        device = torch.device("cuda:" + str(torch.cuda.current_device())) if torch.cuda.is_available() else torch.device("cpu")
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            # print('pred', pred.size())
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            losses = loss(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            if args.opt == 'alig2':
                print('losses', losses.size(), 'idx', idx.size(), 'fhat', optimizer.fhat[idx].size(), 'pred', pred.size())
                input('press any key')
                clipped_losses = torch.max(losses, optimizer.fhat[idx])
                losses = clipped_losses

            loss_value = losses.mean()
            loss_value.backward()
            # print('loss', loss_value)

            if args.opt == 'alig2':
                optimizer.step(lambda: (idx,losses))
            else:
               optimizer.step()

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

        # y_true = batch.y.view(pred.shape).detach().cpu().numpy()
        # y_pred = pred.detach().cpu().numpy()

        # input_dict = {"y_true": y_true, "y_pred": y_pred}
        # roc = evaluator.eval(input_dict)
        # print('roc', roc)

        # monitoring
        # batch_size = y_pred.size(0)
        # xp.train.acc.update(roc[dataset.eval_metric], weighting=batch_size)

        batch_size = len(pred)
        xp.train.loss.update(loss_value, weighting=batch_size)
        xp.train.step_size.update(optimizer.step_size, weighting=batch_size)
        xp.train.step_size_u.update(optimizer.step_size_unclipped, weighting=batch_size)
        xp.train.alpha0.update(optimizer.step_0, weighting=batch_size)
        xp.train.alpha1.update(optimizer.step_1, weighting=batch_size)
        xp.train.alpha2.update(optimizer.step_2, weighting=batch_size)
        xp.train.alpha3.update(optimizer.step_3, weighting=batch_size)
        xp.train.alpha4.update(optimizer.step_4, weighting=batch_size)

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    batch_size = len(y_pred)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    # print('input dic', input_dict)
    roc = evaluator.eval(input_dict)
    # print('roc', roc)
    # for p in model.parameters():
        # print('p',p,'grad',p.grad)
        # input('press any key')

    xp.train.acc.update(roc[dataset.eval_metric], weighting=batch_size)
    # xp.train.grad_norm.update(torch.sqrt(sum(p.grad.data.norm() ** 2  for p in model.parameters())))
    xp.train.weight_norm.update(torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters())))
    xp.train.reg.update(0.5 * args.weight_decay * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.lower_bound.update(optimizer.step_0)
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


def train(model, loss, optimizer, loader, args, xp):
    model.train()

    for metric in xp.train.metrics():
        metric.reset()

    for idx, data in tqdm(loader, disable=not args.tqdm, desc='Train Epoch',
                                               leave=False, total=len(loader)):
        x_, y = data
        if isinstance(x_,dict):
            transforms, x = x_['trans'], x_['image']
        else:
            x = x_
        # print('trans',transforms)
        # input('press any key')
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        # forward pass
        scores = model(x)

        # compute the loss function, possibly using smoothing
        # with set_smoothing_enabled(args.smooth_svm):
        losses = loss(scores, y)
        if args.opt == 'alig2':
            clipped_losses = torch.max(losses, optimizer.fhat[idx])
            losses = clipped_losses

        loss_value = losses.mean()
        print('loss', loss_value)

        # backward pass
        optimizer.zero_grad()
        loss_value.backward()
            # optimization step
        if args.opt == 'alig2':
            optimizer.step(lambda: (idx,losses))
        else:
            optimizer.step(lambda: loss_value)

        if 'sbd' in args.opt and not optimizer.n == 1:
            continue

        # monitoring
        batch_size = x.size(0)
        xp.train.acc.update(accuracy(scores, y), weighting=batch_size)
        xp.train.loss.update(loss_value, weighting=batch_size)
        xp.train.step_size.update(optimizer.step_size, weighting=batch_size)
        xp.train.step_size_u.update(optimizer.step_size_unclipped, weighting=batch_size)
        if args.dataset == "imagenet":
            xp.train.acc5.update(accuracy(scores, y, topk=5), weighting=batch_size)
        xp.train.alpha0.update(optimizer.step_0, weighting=batch_size)
        xp.train.alpha1.update(optimizer.step_1, weighting=batch_size)
        xp.train.alpha2.update(optimizer.step_2, weighting=batch_size)
        xp.train.alpha3.update(optimizer.step_3, weighting=batch_size)
        xp.train.alpha4.update(optimizer.step_4, weighting=batch_size)

    xp.train.grad_norm.update(torch.sqrt(sum(p.grad.data.norm() ** 2  for p in model.parameters())))
    xp.train.weight_norm.update(torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters())))
    xp.train.reg.update(0.5 * args.weight_decay * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.lower_bound.update(optimizer.step_0)
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

def train_old(model, loss, optimizer, loader, args, xp):
    model.train()

    for metric in xp.train.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm, desc='Train Epoch',
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        # if args.opt == 'sbd-sb':
            # # while not (optimizer.n == 1):
            # for _ in range(args.k-1):
                # loss_value, scores = forward_backwards(model, loss, optimizer, x, y)
        # else:
            # loss_value, scores = forward_backwards(model, loss, optimizer, x, y)

        # forward pass
        scores = model(x)
        # compute the loss function, possibly using smoothing
        # with set_smoothing_enabled(args.smooth_svm):
        loss_value = loss(scores, y)
        # backward pass
        optimizer.zero_grad()
        loss_value.backward()
        # optimization step
        optimizer.step(lambda: float(loss_value))

        if 'sbd' in args.opt and not optimizer.n == 1:
            continue

        # monitoring
        batch_size = x.size(0)
        xp.train.acc.update(accuracy(scores, y), weighting=batch_size)
        xp.train.loss.update(loss_value, weighting=batch_size)
        xp.train.step_size.update(optimizer.step_size, weighting=batch_size)
        xp.train.step_size_u.update(optimizer.step_size_unclipped, weighting=batch_size)
        if args.dataset == "imagenet":
            xp.train.acc5.update(accuracy(scores, y, topk=5), weighting=batch_size)

        xp.train.alpha0.update(optimizer.step_0, weighting=batch_size)
        xp.train.alpha1.update(optimizer.step_1, weighting=batch_size)
        xp.train.alpha2.update(optimizer.step_2, weighting=batch_size)
        xp.train.alpha3.update(optimizer.step_3, weighting=batch_size)
        xp.train.alpha4.update(optimizer.step_4, weighting=batch_size)

    xp.train.weight_norm.update(torch.sqrt(sum(p.norm() ** 2 for p in model.parameters())))
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
def test_rank(model, loss, optimizer, loader, args, xp):
    model.eval()
    R = torch.tensor([]).float().cuda()
    R_star = torch.tensor([]).long().cuda()

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
        R_star = torch.cat((R_star,y),0)
        R = torch.cat((R,scores),0)

    loss_val = loss(R, R_star)
    xp_group.acc.update(loss_val)
    xp_group.timer.update()

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.5f}% \t'
          .format(int(xp.epoch.value),
                  tag=loader.tag.title(),
                  timer=xp_group.timer.value,
                  acc=xp_group.acc.value))

    if loader.tag == 'val':
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

@torch.autograd.no_grad()
def test(model, optimizer, loader, args, xp):
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
        if args.dataset == "imagenet":
            xp_group.acc5.update(accuracy(scores, y, topk=5), weighting=scores.size(0))

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
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

@torch.autograd.no_grad()
def test_graph(model, optimizer, evaluator, dataset, loader, args, xp):
    model.eval()
    device = torch.device("cuda:" + str(torch.cuda.current_device())) if torch.cuda.is_available() else torch.device("cpu")

    if loader.tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()

    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    roc = evaluator.eval(input_dict)
    xp_group.acc.update(roc[dataset.eval_metric], weighting=len(y_true))
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
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

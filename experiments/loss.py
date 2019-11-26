import torch.nn as nn
import torch.nn.functional as F
from models.main import get_model, load_best_model
import torch

def get_loss(args):
    if args.teacher:
        if args.decay_lower_bound:
            loss_fn = Distillation_Loss_2(args)
        else:
            loss_fn = Distillation_Loss(args)

    elif args.loss == 'ce':
        loss_fn = CE_Loss()
    else:
        raise ValueError

    print('L2 regularization: \t {}'.format(args.weight_decay))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn

class CE_Loss(nn.Module):

    def __init__(self):
        super(CE_Loss, self).__init__()
        print('creating cross entropy loss')
        self.loss = nn.CrossEntropyLoss()

    def forward(self, scores, y, x):
        return self.loss(scores, y), 0

class Distillation_Loss(nn.Module):

    def __init__(self, args):
        super(Distillation_Loss, self).__init__()

        # print('Creating distillation loss')

        load_model = args.load_model
        # print(args.load_model, load_model)
        # input('press any key')
        model_arch = args.model
        depth = args.depth

        args.load_model = args.teacher
        if 'spiral' in args.teacher:
            args.depth = 5
            args.width = 50
        elif '20' in args.teacher:
            args.depth = 20
        elif '32' in args.teacher:
            args.depth = 32
        elif '44' in args.teacher:
            args.depth = 44
        elif '56' in args.teacher:
            args.depth = 56
        elif '110' in args.teacher:
            args.depth = 110
        else:
            print('unknown teacher architecture')
            raise NotImplementedError

        self.teacher_model = get_model(args)
        self.teacher_model.eval()

        print(args.load_model, load_model)
        # print(args.load_model, load_model)
        args.load_model = load_model
        # print(args.load_model, load_model)
        # input('press any key')
        args.model = model_arch
        args.depth = depth

        self.lambda_t = args.lambda_t
        self.tau = args.tau
        self.lower_bound = args.B

        self.loss = nn.CrossEntropyLoss()

        self.kl = False
        if args.loss == 'kl':
            self.kl = True

    def forward(self, scores, y, x):
        with torch.no_grad():
            scores_teacher = self.teacher_model(x).detach()
        loss_ce = self.loss(scores, y)
        loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) - F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1)
        if self.kl:
            loss = loss_dist.clamp(min=self.lower_bound).mean()
        else:
            # loss = loss_ce.mul(1-self.lambda_t) + loss_dist.mul(self.lambda_t)
            loss = loss_ce.mean() + loss_dist.mul(self.lambda_t).mean()
        return loss, loss_dist.mean()


class Distillation_Loss_2(nn.Module):

    def __init__(self, args):
        super(Distillation_Loss_2, self).__init__()

        print('Creating distillation loss version two')

        load_model = args.load_model
        model_arch = args.model
        depth = args.depth

        args.load_model = args.teacher
        if 'spiral' in args.teacher:
            args.depth = 5
            args.width = 50
        elif '20' in args.teacher:
            args.depth = 20
        elif '32' in args.teacher:
            args.depth = 32
        elif '44' in args.teacher:
            args.depth = 44
        elif '56' in args.teacher:
            args.depth = 56
        elif '110' in args.teacher:
            args.depth = 110
        else:
            print('unknown teacher architecture')
            raise NotImplementedError

        self.teacher_model = get_model(args)
        self.teacher_model.eval()

        print(args.load_model, load_model)
        # print(args.load_model, load_model)
        args.load_model = load_model
        # print(args.load_model, load_model)
        # input('press any key')
        args.model = model_arch
        args.depth = depth

        self.lambda_t = args.lambda_t
        self.tau = args.tau
        self.lower_bound = torch.zeros(arg.train_size).fill_(args.B)
        self.decay_lower_bound = args.decay_lower_bound

        self.loss = nn.CrossEntropyLoss()

        self.kl = False
        if args.loss == 'kl':
            self.kl = True

    def forward(self, scores, y, x, idx):
        with torch.no_grad():
            scores_teacher = self.teacher_model(x).detach()
        loss_ce = self.loss(scores, y)
        loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) - F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1)

        if self.kl:
            lower_bound = self.lower_bound[idx]
            mask = (loss_dist <= lower_bound).long()
            decay_amount = torch.ones_like(loss_dist)
            dacay_amount -= (1-self.decay_lower_bound)*mask
            lower_bound *= decay_amount

            loss = loss_dist.clamp(min=self.lower_bound).mean()
        else:
            loss = loss_ce.mean() + loss_dist.mul(self.lambda_t).mean()

        return loss, loss_dist.mean()
'''
'''

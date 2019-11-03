import torch.nn as nn
import torch.nn.functional as F
from models.main import get_model, load_best_model
import torch

def get_loss(args):
    if args.teacher:
        loss_fn = Distillation_Loss(args)
    else:
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = CE_Loss()

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

        print('Creating distillation loss')

        load_model = args.load_model
        model_arch = args.model
        depth = args.depth

        args.load_model = args.teacher
        if '20' in args.teacher:
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

        args.load_model = load_model
        args.model = model_arch
        args.depth = depth

        self.lambda_t = args.lambda_t
        self.tau = args.tau

        self.loss = nn.CrossEntropyLoss()

    def forward(self, scores, y, x):
        with torch.no_grad():
            scores_teacher = self.teacher_model(x).detach()
        loss_ce = self.loss(scores, y)
        loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) - F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1).mean()
        # loss = loss_ce.mul(1-self.lambda_t) + loss_dist.mul(self.lambda_t)
        loss = loss_ce + loss_dist.mul(self.lambda_t)
        return loss, loss_dist

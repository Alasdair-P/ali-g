import torch.nn as nn
import torch.nn.functional as F
from models.main import get_model, load_best_model
import torch

def get_loss(args):
    if args.teacher:
        loss_fn = Distillation_Loss(args)
    else:
        loss_fn = nn.CrossEntropyLoss()

    print('L2 regularization: \t {}'.format(args.weight_decay))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn

class Distillation_Loss(nn.Module):

    def __init__(self, args):
        super(Distillation_Loss, self).__init__()

        print('Creating distillation loss')

        load_model = args.load_model
        model_name = args.model_name

        args.load_model = args.teacher

        if '18_pretrained' in args.teacher:
            print('Teacher network: ResNet18 pretrained')
            args.model_name = 'ResNet18_pretrained'
            args.load_model = None
        elif '34_pretrained' in args.teacher:
            print('Teacher network: ResNet34pretrained')
            args.model_name = 'ResNet34_pretrained'
            args.load_model = None
        elif '18' in args.teacher:
            print('Teacher network: ResNet18')
            args.model_name = 'ResNet18'
        elif '34' in args.teacher:
            print('Teacher network: ResNet34')
            args.model_name = 'ResNet34'
        else:
            print('unknown teacher architecture')
            raise NotImplementedError

        self.teacher_model = get_model(args)

        args.load_model = load_model
        args.model_name = model_name

        self.lambda_t = args.lambda_t
        self.tau = args.tau
        self.K = args.K

        self.loss = nn.CrossEntropyLoss()

    def forward(self, scores, a_1, a_2, y, x):
        with torch.no_grad():
            scores_teacher, t_a_1, t_a_2 = self.teacher_model(x).detach()

        loss_l2_1 = (t_a_1 - a_1).pow(2).mean()

        loss_l2_2 = (t_a_2 - a_2).pow(2).mean()

        loss_ce = self.loss(scores, y).mean()

        loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) - F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1).mean()

        loss = loss_ce + loss_dist.mul(self.lambda_t) + loss_l2_1.mul(self.K) + loss_l2_2.mul(self.K)

        return loss

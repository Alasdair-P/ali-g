import torch.nn as nn
import torch.nn.functional as F
import torch

def get_model(args):

    if args.model == "ResNet18":
        model = models.resnet18()
    elif args.model == "ResNet18_pretrained":
        model = models.resnet18(pretrained=True)
    elif args.model == "ResNet34":
        model = models.resnet34()
    elif args.model == "ResNet34_pretrained":
        model = models.resnet34(pretrained=True)
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

class Distillation_Loss(nn.Module):

    def __init__(self, params, loss, args):
        """
        Primal Objective with generic loss
        """
        super(Distillation_Loss, self).__init__()

        print('Creating distillation loss')

        load_model = args.load_model
        model_name = args.model_name

        args.load_model = args.teacher
        if '20' in args.teacher:
            print('Teacher network: ResNet20')
            args.model_name = 'ResNet20'
        elif '56' in args.teacher:
            print('Teacher network: ResNet56')
            args.model_name = 'ResNet56'
        elif '110' in args.teacher:
            print('Teacher network: ResNet110')
            args.model_name = 'ResNet110'
        elif '18' in args.teacher:
            print('Teacher network: ResNet18')
            args.model_name = 'ResNet18'
        elif '18_pretrained' in args.teacher:
            print('Teacher network: ResNet18 pretrained')
            args.model_name = 'ResNet18_pretrained'
            args.load_model = None
        elif '34_pretrained' in args.teacher:
            print('Teacher network: ResNet34pretrained')
            args.model_name = 'ResNet34_pretrained'
            args.load_model = None
        else:
            print('unknown teacher architecture')
            assert 2==1
        self.teacher_model = get_model(args)

        args.load_model = load_model
        args.model_name = model_name

        self.param_groups = params
        self.loss = loss
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.distill_loss = args.distill_loss
        print('distill loss :', self.distill_loss)
        self.epoch = 0
        self.lambda_t = args.lambda_t
        self.tau = args.tau
        self.hq_epoch = args.hq_epoch

    def regularization_l1(self):
        reg = 0
        for group in self.param_groups:
            l1 = group['l1']
            if not l1:
                continue
            for p in group['params']:
                reg += l1 * (p - BinarizeWeightMirrorDescent.apply(p).float()).norm(1)
        return reg

    def regularization_l2(self):
        reg = 0
        for group in self.param_groups:
            l2 = group['weight_decay']
            if not l2:
                continue
            for p in group['params']:
                reg += 0.5 * l2 * p.norm() ** 2
        return reg

    def epoch_update(self):
        self.epoch += 1
        if self.epoch < self.hq_epoch:
            self.lambda_t -= (self.tau - 1).div(self.hq_epoch)

    def forward(self, scores, y, x):
        with torch.no_grad():
            scores_teacher = self.teacher_model(x).detach()

        loss_ce = self.loss(scores, y).mean()

        # targerts = torch.zeros_like(scores)
        # targerts[torch.arange(len(y)),y] = 1

        # correct
        if self.distill_loss == 'ce':
            loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) )).sum(dim=1).mean()
        elif self.distill_loss == 'kl':
            loss_dist = -(F.softmax(scores_teacher.div(self.tau),dim=1).mul( F.log_softmax(scores.div(self.tau),dim=1) - F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1).mean()
        elif self.distill_loss == 'ce_wrong':
            loss_dist = -(F.softmax(scores.div(self.tau),dim=1).mul( F.log_softmax(scores_teacher.div(self.tau),dim=1) )).sum(dim=1).mean()
        else: #wrong
            print('loss type unknown')
            assert 2 == 1

        # loss_dist = -(F.softmax(scores_teacher,dim=1).mul( F.log_softmax(scores,dim=1) - F.log_softmax(scores_teacher,dim=1) )).sum(dim=1).mean()
        # loss_dist = self.kl_loss(F.log_softmax(scores, dim=1), F.softmax(scores_teacher,dim=1))

        loss = loss_ce + loss_dist.mul(self.lambda_t)

        return loss, loss_ce, loss_dist

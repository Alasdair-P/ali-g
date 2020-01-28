import torch.optim

# from dfw import DFW
# from dfw.baselines import BPGrad
# from l4pytorch import L4Mom, L4Adam
from alig.th import AliG
# from alig.th.projection import l2_projection
from cgd import CGD
from segd import SEGD
from sbd import SBD
from sbd import SBD
from sbdf import SBDF
from sbd_backward import SBDB
from segd3 import SEGD3
from alig_w_segd import ALIG_SEGD

@torch.autograd.no_grad()
def l2_projection(parameters, max_norm):
    if max_norm is None:
        return
    total_norm = torch.sqrt(sum(p.norm() ** 2 for p in parameters))
    if total_norm > max_norm:
        ratio = max_norm / total_norm
        for p in parameters:
            p *= ratio

def get_optimizer(args, model, loss, parameters):
    parameters = list(parameters)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay, amsgrad=True)
    # elif args.opt == 'dfw':
        # optimizer = DFW(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.opt == 'bpgrad':
        # optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'alig':
        optimizer = AliG(parameters, max_lr=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), lower_bound=args.B)
    elif args.opt == 'cgd':
        optimizer = CGD(parameters, model, loss, eta=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), debug=args.debug, eps=args.fd)
    elif args.opt == 'segd':
        optimizer = SEGD(parameters, model, loss, eta=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    elif args.opt == 'alig_segd':
        optimizer = ALIG_SEGD(parameters, model, loss, eta=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    elif args.opt == 'segd3':
        optimizer = SEGD3(parameters, model, loss, eta=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    elif args.opt == 'sbd':
        optimizer = SBD(parameters, model, loss, eta=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    elif args.opt == 'sbdf':
        optimizer = SBDF(parameters, model, loss, eta_2=args.eta_2 or args.eta, zero_plane=args.zero_plane, eta=args.eta, n=args.n, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    elif args.opt == 'sbdb':
        optimizer = SBDB(parameters, model, loss, eta=args.eta, n=args.n, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), weight_decay=args.weight_decay)
    # elif args.opt == 'bpgrad':
        # optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.opt == 'l4adam':
        # optimizer = L4Adam(parameters, weight_decay=args.weight_decay)
    # elif args.opt == 'l4mom':
        # optimizer = L4Mom(parameters, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.step_size = args.eta
    optimizer.step_size_unclipped = args.eta
    optimizer.momentum = args.momentum
    optimizer.step_0 = 0
    optimizer.step_1 = 0
    optimizer.step_2 = 0
    optimizer.step_3 = 0
    optimizer.step_4 = 0

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer

def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor

        optimizer.step_size = optimizer.param_groups[0]['lr']
        optimizer.step_size_unclipped = optimizer.param_groups[0]['lr']
        print('decaying learning rate to:', optimizer.step_size )
    else:
        raise ValueError

def decay_lower_bound(args, loss, optimizer, model):
    filename = '{0}/model_lb_{1:.6f}.pkl'.format(args.xp_name, args.B)
    filename.replace('.', '_')
    save_state(model, optimizer, filename)
    print('saving model to {}'.format(filename))
    args.B *= args.decay_lower_bound
    if args.opt == 'alig':
        optimizer.B = args.B
    loss.lower_bound = args.B
    print('decaying lower boaund to:', args.B)

def reset_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.eta_initial
    optimizer.step_size = optimizer.param_groups[0]['lr']
    optimizer.step_size_unclipped = optimizer.param_groups[0]['lr']
    print('updatign T to:', args.T)

def decay_stuff(xp, model, args, optimizer, loss, i):
    max_epochs = args.max_epochs
    if max_epochs:
        if xp.train.obj.value / (args.B + 1e-9) <= (1 + 1e-3) or args.i % max_epochs == (max_epochs - 1):
            args.i = 0
            if args.decay_lower_bound:
                decay_lower_bound(args, loss, optimizer, model)
            if isinstance(optimizer, torch.optim.SGD):
                reset_lr(args, optimizer)
        else:
            args.i += 1

        if args.i in args.T:
            decay_optimizer(optimizer, args.decay_factor)
    else:
        if i in args.T:
            decay_optimizer(optimizer, args.decay_factor)

def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

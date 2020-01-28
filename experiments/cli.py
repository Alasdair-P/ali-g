import os
import argparse
import warnings
import copy

from cuda import set_cuda

def parse_command():
    parser = argparse.ArgumentParser()

    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_optimization_parser(parser)
    _add_loss_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()
    filter_args(args)

    return args

def _add_dataset_parser(parser):
    d_parser = parser.add_argument_group(title='Dataset parameters')
    d_parser.add_argument('--dataset', type=str, default=None,
                          help='dataset')
    d_parser.add_argument('--train_size', type=int, default=None,
                          help="training data size")
    d_parser.add_argument('--val_size', type=int, default=None,
                          help="val data size")
    d_parser.add_argument('--test_size', type=int, default=None,
                          help="test data size")
    d_parser.add_argument('--noise', type=float, default=0,
                          help="noise for spiral data set")
    d_parser.add_argument('--no_data_augmentation', dest='augment',
                          action='store_false', help='no data augmentation')
    d_parser.add_argument('--no_shuffling', dest='no_shuffle',
                          action='store_true', help='no shuffling')
    d_parser.set_defaults(augment=True)

def _add_model_parser(parser):
    m_parser = parser.add_argument_group(title='Model parameters')
    m_parser.add_argument('--model', type=str,
                          help="model name")
    m_parser.add_argument('--depth', type=int, default=None,
                          help="depth of network on densenet / wide resnet")
    m_parser.add_argument('--width', type=int, default=None,
                          help="width of network on wide resnet")
    m_parser.add_argument('--growth', type=int, default=None,
                          help="growth rate of densenet")
    m_parser.add_argument('--no_bottleneck', dest="bottleneck", action="store_false",
                          help="bottleneck on densenet")
    m_parser.add_argument('--dropout', type=float, default=0,
                          help="dropout rate")
    m_parser.add_argument('--active_func', type=str, default='relu',
                          help="activation function")
    m_parser.add_argument('--load_model', '--load-model', dest='load_model', default=None,
                          help='data file with model')
    m_parser.set_defaults(pretrained=False, wrn=False, densenet=False, bottleneck=True)

def _add_optimization_parser(parser):
    o_parser = parser.add_argument_group(title='Training parameters')
    o_parser.add_argument('--epochs', type=int, default=None,
                          help="number of epochs")
    o_parser.add_argument('--batch_size', '--batch-size', '--b', dest='batch_size', type=int, default=None,
                          help="batch size")
    o_parser.add_argument('--eta', type=float, default=0.1,
                          help="initial / maximal learning rate")
    o_parser.add_argument('--eta_2', type=float, default=None,
                          help="initial / maximal learning rate")
    o_parser.add_argument('--momentum', type=float, default=0.9,
                          help="momentum value for SGD")
    o_parser.add_argument('--opt', type=str, required=True,
                          help="optimizer to use")
    o_parser.add_argument('--T', type=int, default=[-1], nargs='+',
                          help="number of epochs between proximal updates / lr decay")
    o_parser.add_argument('--decay_factor', '--decay-factor', dest='decay_factor', type=float, default=0.1,
                          help="decay factor for the learning rate / proximal term")
    o_parser.add_argument('--load_opt', default=None,
                          help='data file with opt' )
    o_parser.add_argument('--temp_rate', type=float, default=1e-6,
                          help="rate at which temperature is increase")
    o_parser.add_argument('--hq_epoch', type=int, default=-1,
                          help="hard qunatisation epoch")
    o_parser.add_argument('--fd', type=float, default=1e-2,
                          help="finite difference")
    o_parser.add_argument('--max_epochs', type=int, default=0,
                          help="max epochs before decaying lower bound")
    o_parser.add_argument('--n', type=int, default=1,
                          help="size of bundle")
    o_parser.add_argument('--no_zero_plane', dest='zero_plane', action='store_false',
                          help="to not include alig zero plane in bundle")

def _add_loss_parser(parser):
    l_parser = parser.add_argument_group(title='Loss parameters')
    l_parser.add_argument('--weight_decay', type=float, default=0,
                          help="l2-regularization")
    l_parser.add_argument('--max_norm', type=float, default=None,
                          help="maximal l2-norm for constrained optimizers")
    l_parser.add_argument('--loss', type=str, default='ce', choices=("svm", "ce", "distill", "kl"),
                          help="loss function to use ('svm' or 'ce')")
    l_parser.add_argument('--smooth_svm', dest="smooth_svm", action="store_true",
                          help="smooth SVM")
    l_parser.add_argument('--teacher', dest="teacher", type=str, default=None,
                          help="path to teacher model")
    l_parser.add_argument('--tau', type=float, default=3,
                          help="distill loss smoothing")
    l_parser.add_argument('--lambda_t', type=float, default=10,
                          help="distll loss strenght")
    l_parser.add_argument('--B', type=float, default=0,
                          help="lower bound on loss")
    l_parser.add_argument('--decay_lower_bound', type=float, default=0,
                          help="factor to decay lower bound by")
    l_parser.set_defaults(smooth_svm=False)

def _add_misc_parser(parser):
    m_parser = parser.add_argument_group(title='Misc parameters')
    m_parser.add_argument('--seed', type=int, default=None,
                          help="seed for pseudo-randomness")
    m_parser.add_argument('--cuda', type=int, default=1,
                          help="use cuda")
    m_parser.add_argument('--no_visdom', dest='visdom', action='store_false',
                          help='do not use visdom')
    m_parser.add_argument('--server', type=str, default='http://helios',
                          help="server for visdom")
    m_parser.add_argument('--port', type=int, default=9020,
                          help="port for visdom")
    m_parser.add_argument('--xp_name', "--xp-name", dest="xp_name", type=str, default=None,
                          help="name of experiment")
    m_parser.add_argument('--xp_dir', type=str,
                          default='/data0/binary-networks-data',
                          help="data root directory for experiments")
    m_parser.add_argument('--no_log', dest='log', action='store_false',
                          help='do not log results')
    m_parser.add_argument('--debug', dest='debug', action='store_true',
                          help='debug mode')
    m_parser.add_argument('--parallel_gpu', dest='parallel_gpu', action='store_true',
                          help="parallel gpu computation")
    m_parser.add_argument('--no_tqdm', dest='tqdm', action='store_false',
                          help="use of tqdm progress bars")
    m_parser.add_argument('--run_no', type=str, default='0',
                          help="which run for repeats")
    m_parser.add_argument('--reg', action='store_true',
                          help="add proxquant style reg")
    m_parser.set_defaults(visdom=True, log=True, debug=False, parallel_gpu=False, tqdm=True)

def set_xp_name(args):
    if args.dataset == 'spiral':
        args.model = 'mlp_d_' + str(args.depth) + '_w_' + str(args.width)
    if args.debug:
        args.log = args.visdom = False
        args.xp_name = '../debug'
        if not os.path.exists(args.xp_name):
            os.makedirs(args.xp_name)
    elif args.xp_name is None:
        xp_name = '../results/{data}/'.format(data=args.dataset)
        xp_name += "{model}{data}-{opt}--eta-{eta}--l2-{l2}--b-{b}--run-{run}"
        l2 = args.max_norm if args.opt == 'alig' else args.weight_decay
        data = args.dataset.replace("cifar", "")
        args.xp_name = xp_name.format(model=args.model, data=data, opt=args.opt, eta=args.eta, l2=args.max_norm or args.weight_decay, b=args.batch_size, run=args.run_no)

    if args.log:
        # generate automatic experiment name if not provided
        args.xp_name = os.path.join(args.xp_dir, args.xp_name)
        if os.path.exists(args.xp_name):
            warnings.warn('An experiment already exists at rm -r {}'
                          .format(os.path.abspath(args.xp_name)))
            raise ValueError
        else:
            os.makedirs(args.xp_name)

    if args.load_model:
        args.load_model = os.path.join(args.xp_dir, args.load_model)

def set_num_classes(args):
    if args.dataset == 'spiral':
        args.n_classes = 2
    elif args.dataset == 'cifar10':
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        args.n_classes = 100
    elif args.dataset == 'snli':
        args.n_classes = 3
    elif 'svhn' in args.dataset:
        args.n_classes = 10
    elif args.dataset == 'imagenet':
        args.n_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        args.n_classes = 200
    else:
        raise ValueError

def set_visdom(args):
    print(args.visdom)
    if not args.visdom:
        return
    if args.server is None:
        if 'VISDOM_SERVER' in os.environ:
            print('visdom server', os.environ['VISDOM_SERVER'])
            args.server = os.environ['VISDOM_SERVER']
        else:
            args.visdom = False
            print("Could not find a valid visdom server, de-activating visdom...")

def filter_args(args):
    args.T = list(args.T)
    args.eta_initial = copy.copy(args.eta)
    args.i = 0
    set_cuda(args)
    set_xp_name(args)
    set_visdom(args)
    set_num_classes(args)

from data.loaders import loaders_cifar, loaders_mnist, loaders_svhn, loaders_imagenet, loaders_mol, loaders_code, loaders_tiny_imagenet

def get_data_loaders(args):

    print('Dataset: \t {}'.format(args.dataset.upper()))

    # remove values if None
    for k in ('train_size', 'val_size', 'test_size'):
        if args.__dict__[k] is None:
            args.__dict__.pop(k)

    if args.dataset == 'mnist':
        loader_train, loader_val, loader_test = loaders_mnist(**vars(args))
    elif 'cifar' in args.dataset:
        loader_train, loader_val, loader_test = loaders_cifar(**vars(args))
    elif 'svhn' in args.dataset:
        loader_train, loader_val, loader_test = loaders_svhn(**vars(args))
    elif 'mol' in args.dataset:
        loader_train, loader_val, loader_test = loaders_mol(**vars(args))
    elif 'code' in args.dataset:
        loader_train, loader_val, loader_test = loaders_code(**vars(args))
    elif args.dataset == 'imagenet':
        loader_train, loader_val, loader_test = loaders_imagenet(**vars(args))
    else:
        raise NotImplementedError

    args.train_size = len(loader_train.dataset)
    args.val_size = len(loader_val.dataset)
    args.test_size = len(loader_test.dataset)
    print('train size: ', args.train_size,'val size: ', args.val_size, 'test size: ', args.test_size)

    return loader_train, loader_val, loader_test

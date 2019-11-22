import torch
import numpy as np
from scipy.stats import multivariate_normal as mnorm
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.autograd import Variable
import imageio
import torch.utils.data as data
import visdom

def get_raw_data(args):
    if args.dataset == 'spiral':
        gen = generate_spiral_data

    np.random.seed(args.seed)
    x_train, y_train = gen(args.train_size, args.noise)
    x_test, y_test = gen(args.test_size, args.noise)
    return x_train.cuda(), y_train.cuda()

def get_data(args):
    if args.dataset == 'spiral':
        gen = generate_spiral_data

    np.random.seed(args.seed)
    x_train, y_train = gen(args.train_size)
    x_test, y_test = gen(args.test_size)

    dataset_train = data.TensorDataset(x_train, y_train)
    dataset_test = data.TensorDataset(x_test, y_test)

    loader_train = data.DataLoader(dataset_train, shuffle=True,
                                   batch_size=args.batch_size)
    loader_test = data.DataLoader(dataset_test, shuffle=False,
                                  batch_size=len(dataset_test))

    return loader_train, loader_test

def generate_spiral_data(size, noise=0):

    n = size // 2  # nb negative examples
    p = size - n  # nb positive examples

    theta1 = np.linspace(0.5, 4 * np.pi, n)
    x1 = theta1[:, None] * np.stack((np.cos(theta1), np.sin(theta1))).T

    theta2 = np.linspace(0.5, 4 * np.pi, p)
    x2 = theta2[:, None] * np.stack((np.cos(np.pi + theta2),
                                     np.sin(np.pi + theta2))).T

    y1 = np.zeros(len(x1), dtype=np.int64)
    y2 = np.ones(len(x2), dtype=np.int64)

    x = np.copy(np.concatenate((x1, x2)))
    y = np.copy(np.concatenate((y1, y2)))

    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    x_noise = torch.randn(x.size()) * noise
    x += x_noise

    return x, y

def plot_decsion_boundary(model, args):
    if 'spiral' in args.dataset:
        visualize_features(model, args)

def visualize_features(model, args):

    env_name = args.xp_name.split('/')[-1]
    visdom_opts={'server': 'http://helios.robots.ox.ac.uk',
                 'port'  : args.port,
                 'env'   : '{}'.format(env_name)}
    vis = visdom.Visdom(**visdom_opts)

    x, y = get_raw_data(args)

    x = Variable(x)
    features = model.features(x).data.cpu().numpy()

    x = x.data.cpu().numpy()
    y = y.cpu().numpy()

    plt.figure(figsize=(12, 12))

    # mesh grid resolution
    h = 0.01
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cmap = 'coolwarm'

    XX = Variable(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).cuda()
    out = model(XX).cpu()
    ZZ = nn.functional.softmax(out,dim=1).data.numpy()[:, 0]
    ZZ = 1 - ZZ.reshape(xx.shape)
    plt.subplot(2, 1, 1)
    plt.axis('equal')
    plt.contourf(xx, yy, ZZ, cmap=cmap, alpha=0.8)

    indices = np.where(y == 0)
    plt.plot(x[indices, 0], x[indices, 1], 'bo', alpha=0.5)
    indices = np.where(y == 1)
    plt.plot(x[indices, 0], x[indices, 1], 'ro', alpha=0.5)
    plt.ylabel('Original Space')

    plt.subplot(2, 1, 2)
    # plt.axis('equal')
    indices = np.where(y == 0)
    plt.plot(features[indices, 0], features[indices, 1], 'bo', alpha=0.5)
    indices = np.where(y == 1)
    plt.plot(features[indices, 0], features[indices, 1], 'ro', alpha=0.5)

    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    XX = Variable(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).cuda()
    ZZ = nn.functional.softmax(model.clf(XX), dim=1).data.cpu().numpy()[:, 0]
    ZZ = 1 - ZZ.reshape(xx.shape)
    plt.contourf(xx, yy, ZZ, cmap=cmap, alpha=0.8)
    plt.axis('equal')
    plt.ylabel('Latent Space')

    plt.savefig(        '{}.png'.format(args.xp_name))
    im = imageio.imread('{}.png'.format(args.xp_name))
    im = im[:,:,0:3]
    im = np.transpose(im, (2,0,1))
    vis.image(im,
                win='decision boundary',
                opts=dict(caption='decision boundary', title='decision boundary'))

if __name__=="__main__":
    import imageio
    visdom_opts={'server': 'http://helios.robots.ox.ac.uk',
                 'port'  : 9020,
                 'env'   : '{}'.format('test')}
    plt.axis('equal')
    vis = visdom.Visdom(**visdom_opts)
    xx = np.tile(np.arange(1, 101), (100, 1))
    yy = xx.transpose()
    ZZ = np.exp((((xx - 50) ** 2) + ((yy - 50) ** 2)) / -(20.0 ** 2))
    cmap = 'coolwarm'
    plt.contourf(xx, yy, ZZ, cmap=cmap, alpha=0.8)
    plt.savefig('test.png')
    im = imageio.imread('test.png')
    print(im.shape)
    im = im[:,:,0:3]
    print(im.shape)
    im = np.transpose(im, (2,0,1))
    print(im.shape)
    vis.image(im,
                win='decision boundary',
                opts=dict(caption='decision boundary', title='decision boundary'))

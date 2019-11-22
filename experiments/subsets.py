from cli import parse_command
from cuda import set_cuda
from data import get_data_loaders
from tqdm import tqdm
from models.main import get_model, load_best_model
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import scipy.stats
import visdom

import torch

def main(args):

    visdom_opts={'server': 'http://helios.robots.ox.ac.uk',
                 'port'  : 9012,
                 'env'   : '{}'.format('view_data')}
    vis = visdom.Visdom(**visdom_opts)

    batch_size = args.batch_size
    set_cuda(args)

    args.model_name = 'resnet'
    args.depth = 32
    args.load_model = '/data0/binary-networks-data/cifar100/cifar100_resnet32/model.pkl'
    model_teacher = get_model(args)

    args.model_name = 'resnet'
    args.depth = 20
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.500000.pkl'
    # B = 0.5
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.400000.pkl'
    # B = 0.4*0.8**0
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.320000.pkl'
    # B = 0.4*0.8**1
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.256000.pkl'
    # B = 0.4*0.8**2
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.204800.pkl'
    # B = 0.4*0.8**3
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.163840.pkl'
    # B = 0.4*0.8**4
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.131072.pkl'
    # B = 0.4*0.8**5
    # args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.104858.pkl'
    # B = 0.4*0.8**6
    args.load_model = '/data0/results/cifar100/resnet100-sgd--eta-1.0--l2-0--b-128--run-teacher32_automatic_lower_bound_reduction_0.5_no_data_augmentation_2/model_lb_0.067109.pkl'
    B = 0.4*0.8**7
    model_student = get_model(args)


    dists_student = np.zeros((args.train_size,args.n_classes))
    dists_teacher = np.zeros((args.train_size,args.n_classes))
    true_class = np.zeros((args.train_size,args.n_classes))
    correct_student = np.zeros((args.train_size))
    correct_teacher = np.zeros((args.train_size))
    entropy_student = np.zeros((args.train_size))
    entropy_teacher = np.zeros((args.train_size))
    cross_entropy_student = np.zeros((args.train_size))
    cross_entropy_teacher = np.zeros((args.train_size))
    kl = np.zeros((args.train_size))
    kl_ = np.zeros((args.train_size, args.n_classes))

    model_student.eval()
    model_teacher.eval()

    start_idx = 0

    args.shuffling = 'none'
    loader_train, loader_val, loader_test = get_data_loaders(args)

    for idx, (x, y) in tqdm(enumerate(loader_train), disable=not args.tqdm, desc='Train Epoch',
                            leave=False, total=len(loader_train)):

        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        with torch.no_grad():

            scores_student = model_student(x)
            scores_teacher = model_teacher(x)

            max_scores_student = scores_student.argmax(dim=1)
            max_scores_teacher = scores_teacher.argmax(dim=1)
            correct_s = max_scores_student.eq(y).long()
            correct_t = max_scores_teacher.eq(y).long()

            # print(correct_s, max_scores_student, y)
            # input('press any key')
            one_hot = torch.zeros((args.batch_size, args.n_classes)).float().cuda()
            one_hot[torch.arange(args.batch_size),y] = 1

            dist_teacher = F.softmax(scores_teacher.div(args.tau),dim=1)
            dist_student = F.softmax(scores_student.div(args.tau),dim=1)

            loss_dist = -(F.softmax(scores_teacher.div(args.tau),dim=1).mul( F.log_softmax(scores_student.div(args.tau),dim=1) - F.log_softmax(scores_teacher.div(args.tau),dim=1) )).sum(dim=1)
            ce_student = -one_hot.mul( F.log_softmax(scores_student,dim=1) ).sum(dim=1)
            ce_teacher = -one_hot.mul( F.log_softmax(scores_teacher,dim=1) ).sum(dim=1)
            entropy_t = -dist_teacher.mul(dist_teacher.log()).sum(dim=1)
            entropy_s = -dist_student.mul(dist_student.log()).sum(dim=1)

            # loss_dist *= (1-correct_s.float())

            dists_student[start_idx:start_idx+batch_size,:] = dist_student.cpu().numpy()
            dists_teacher[start_idx:start_idx+batch_size,:] = dist_teacher.cpu().numpy()

            entropy_student[start_idx:start_idx+batch_size] = entropy_s.cpu().numpy()
            entropy_teacher[start_idx:start_idx+batch_size] = entropy_t.cpu().numpy()

            cross_entropy_student[start_idx:start_idx+batch_size] = entropy_s.cpu().numpy()
            cross_entropy_teacher[start_idx:start_idx+batch_size] = entropy_t.cpu().numpy()

            correct_student[start_idx:start_idx+batch_size] = correct_s.cpu().numpy()
            correct_teacher[start_idx:start_idx+batch_size] = correct_t.cpu().numpy()

            true_class[start_idx:start_idx+batch_size,:] = one_hot.cpu().numpy()

            kl[start_idx:start_idx+batch_size] = loss_dist.cpu().numpy()
            kl_[start_idx:start_idx+batch_size, 0] = loss_dist.cpu().numpy()

            start_idx += batch_size

    print('acc teacher', correct_teacher.mean())
    print('acc student', correct_student.mean())
    print('kl', kl.mean())
    print('lower bound', B)
    print('entropy_student', entropy_student.mean())
    print('entropy_teacher', entropy_teacher.mean())
    # Filter stuff
    # set corret = 1 to view correct examples sets correct = 0 to view incorrect exampels
    filter_type = 'all'
    # filter_type = 'student'
    # filter_type = 'teacher'
    correct = 0
    if filter_type == 'student':
        entropy_student = entropy_student[correct_student==correct]
        entropy_teacher = entropy_teacher[correct_student==correct]
        kl = kl[correct_student==correct]
        correct_student = correct_student[correct_student==correct]
    elif filter_type == 'teacher':
        entropy_student = entropy_student[correct_teacher==correct]
        entropy_teacher = entropy_teacher[correct_teacher==correct]
        kl = kl[correct_teacher==correct]
        correct_student = correct_student[correct_teacher==correct]


    indexs = np.argsort(kl)
    num = 10

    true_100 = true_class[indexs[-num:],:]
    st_100 = dists_student[indexs[-num:],:]
    te_100 = dists_teacher[indexs[-num:],:]
    kl_100 = kl_[indexs[-num:],:]

    # true_100 = true_class[indexs[:num],:]
    # st_100 = dists_student[indexs[:num],:]
    # te_100 = dists_teacher[indexs[:num],:]

    mask = kl > B
    print('number failed to reach lower bound', mask.sum())
    print('kl highest', kl[indexs[-num:]])
    print('kl loweest', kl[indexs[:num]])
    print('indexs', np.arange(args.train_size)[mask])
    filename = '/data0/results/cifar100/numpy_arrays/' + args.load_model[134:-4]
    print(filename)
    filename = filename.replace('.','_')
    print(filename)
    filename = filename + '.npy'
    np.save(filename, np.arange(args.train_size)[mask])
    input('press any key')

    kl_es_st = np.concatenate((kl.reshape(-1,1), entropy_student.reshape(-1,1), entropy_teacher.reshape(-1,1)), axis=1)
    kl_ces_cst = np.concatenate((kl.reshape(-1,1), cross_entropy_student.reshape(-1,1), cross_entropy_teacher.reshape(-1,1)), axis=1)
    vis.scatter(
            # X=kl_es_st,
            X=kl_ces_cst,
            # Y=(correct_teacher+1).astype(int),
            # Y=(correct_student+1).astype(int),
            Y=(mask+1).astype(int),
            opts=dict(
                # legend=['Men', 'Women'],
                markersize=2,
                xtickmin=0,
                xtickmax=2,
                xlabel='kl',
                xtickstep=0.5,
                ytickmin=0,
                ytickmax=2,
                ytickstep=0.5,
                ylabel='entropy_student',
                ztickmin=0,
                ztickmax=2,
                ztickstep=0.5,
                zlabel='entropy_teacher',
            )
    )

    if False:
        np.savetxt('kl_es_et', kl_es_st)
        np.save('kl_es_et', kl_es_st)

    # print(indexs)
    # print(kl)
    # print(kl[indexs])
    # print(kl[indexs[-num:]])

    """
    kl_es_st = np.concatenate((entropy_student.reshape(-1,1), entropy_teacher.reshape(-1,1)), axis=1)
    vis.scatter(
            X=kl_es_st,
            Y=(correct_teacher+1).astype(int),
            opts=dict(
                # legend=['Men', 'Women'],
                markersize=2,
                xtickmin=0,
                xtickmax=2,
                xtickstep=0.5,
                ytickmin=0,
                ytickmax=2,
                ytickstep=0.5,
                xlabel='entropy_student',
                ylabel='entropy_teacher',
            )
    )

    for i in range(num):
        X_=np.concatenate((true_100[i,:].reshape(-1,1), st_100[i,:].reshape(-1,1), te_100[i,:].reshape(-1,1), kl_100[i,:].reshape(-1,1)),axis=1)
        # print(X_)
        # print(X_.shape)
        vis.bar(
            X=X_,
                    opts=dict(stacked=False,
                                    legend=['true', 'student', 'teacher', 'kl']
                                )
                )
    # vis.bar(X=kl[indexs])

    """
if __name__ ==  '__main__':
    args = parse_command()
    main(args)

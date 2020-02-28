import numpy as np

lb_0_500000 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_500000.npy')
lb_0_400000 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_400000.npy')
lb_0_320000 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_320000.npy')
lb_0_256000 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_256000.npy')
lb_0_204800 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_204800.npy')
lb_0_163840 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_163840.npy')
lb_0_131072 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_131072.npy')
lb_0_104858 = np.load('/data0/results/cifar100/numpy_arrays/model_lb_0_104858.npy')

l = [lb_0_500000, lb_0_400000, lb_0_320000, lb_0_256000, lb_0_204800, lb_0_163840, lb_0_131072, lb_0_104858]
confusion_mat = np.zeros((len(l),len(l)))

for data_index in range(45000):
    for j, array_1 in enumerate(l):
        for k, array_2 in enumerate(l):
            if data_index in array_1 and data_index not in array_2:
                confusion_mat[j,k] += 1
print(l)
for array in l:
    print(len(array))
print(confusion_mat)

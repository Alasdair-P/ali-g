
import torch
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import required

def generate_perms(n):
    # we're creating the binary representation for all numbers from 0 to N-1
    N = 2**n
    # for that, we need a 1xN matrix of all the numbers
    a = np.arange(N, dtype=int).reshape(1,-1)
    # we also need a log2(N)x1 matrix, for the powers of 2 on the numbers.
    # floor(log(N)) is the largest component that can make up any number up to N
    l = int(np.log2(N))
    b = np.arange(l, dtype=int)[::-1,np.newaxis]
    # This step is a bit complicated, so I'll explain it below.
    comb = torch.Tensor(np.array(a & 2**b > 0, dtype=int))
    comb = comb[:,1:]
    return comb

if __name__ == "__main__":
    print(generate_perms(4))

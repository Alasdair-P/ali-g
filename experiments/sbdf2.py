import torch
import math
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import required

class SBDF(optim.Optimizer):
    def __init__(self, params, model, obj, eta_2, eta=None, n=1, momentum=0, projection_fn=None, weight_decay=0, eps=1e-8, adjusted_momentum=False, betas=(0.0, 0.99), amsgrad=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay, betas=betas, amsgrad=amsgrad, eps=eps)
        super(SBDF, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn

        self.print = True
        self.print = False

        self.n = n
        self.K = n
        self.loop_range = self.n
        self.zero_plane = True
        if self.zero_plane:
            self.n += 1
        self.eps = eps
        self.eta_2 = eta_2

        self.reset_bundle()

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
                    self.state[p]['fast_momentum_buffer'] = state['momentum_buffer'].clone()

        self.apply_momentum = self.apply_momentum_nesterov

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def step(self, loss):

        self.update_bundle()

        if self.k == self.K:

            self.construct_A_and_b()

            self.generate_combs()

            self.loop_over_combs()

            self.update_parameters()

            if self.projection is not None:
                self.projection()

            self.reset_bundle()

        else:

            self.sgd_step()

    @torch.autograd.no_grad()
    def reset_bundle(self):
        self.k = 0
        self.losses = []
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'] = []
                self.state[p]['w'] = []
                self.state[p]['fast_momentum_buffer'] = state['momentum_buffer'].clone()

    @torch.autograd.no_grad()
    def update_bundle(self):
        self.k += 1
        self.losses.append(float(loss()))
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grads'].append(p.grad.data.detach().clone())
                state['w'].append(p.data.detach().clone())

    @torch.autograd.no_grad()
    def sgd_step(self):
        for group in self.param_groups:
            for p in group['params']:
                p.add_(-self.eta_2, p.grad.data)
                if group["momentum"]:
                    buffer = self.state[p]['fast_momentum_buffer']
                    momentum = group["momentum"]
                    buffer.mul_(momentum).add_(-self.eta_2, p.grad.data)
                    p.data.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def construct_A_and_b(self):
        """
        construst A matrix and b vector here the zero index is the standard gradient
        the one index is the extra gradient and the 2 index is the ground plane
        """
        self.A = torch.ones(self.n + 1, self.n + 1, device='cuda')
        self.b = torch.zeros(self.n + 1, device='cuda')
        self.A[0:-1,0:-1] = 0
        self.A += self.eps * torch.eye(self.n + 1, device='cuda')
        self.A[-1,-1] = 0
        self.b[-1] = 1

        loop_range = len(self.losses)
        for i in range(loop_range):
            self.b[i] += self.losses[i]

        if self.print:
            print('A')
            print(self.A)
            print('b')
            print(self.b)

        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                for i in range(loop_range):
                    g_i = self.state[p]['grads'][i]
                    w_s = self.state[p]['w']
                    self.b[i] -= g_i.mul(w_s[i]-w_s[0]).sum()
                    for j in range(loop_range):
                        g_j = self.state[p]['grads'][j]
                        self.A[i, j] += eta * (g_i * g_j).sum()

        # self.alig_step = (self.b[0]/self.A[0,0]).clamp(min=0, max=1)
        # self.segd_step = ((self.A[0,0] - self.b[0] + self.b[1] - self.A[0,1]) / (self.A[0,0] - 2 * self.A[0,1] + self.A[1,1])).clamp(min=0, max=1) # calcuates v

        if self.print:
            print('A')
            print(self.A)
            print('b')
            print(self.b)

    @torch.autograd.no_grad()
    def generate_combs(self):
        # we're creating the binary representation for all numbers from 0 to N-1
        N = 2**self.n
        a = np.arange(N, dtype=int).reshape(1,-1)
        l = int(np.log2(N))
        b = np.arange(l, dtype=int)[::-1,np.newaxis]
        self.combinations = torch.Tensor(np.array(a & 2**b > 0, dtype=int)).bool().cuda()
        self.combinations = self.combinations[:,1:]
        # add line adding columns of combs and removing columns thats sum to one
        self.num_combs = self.combinations.size()[1]
        if self.print:
            print('self.combinations')
            print(self.combinations)

    @torch.autograd.no_grad()
    def loop_over_combs(self):
        self.best_alphas = torch.zeros(self.n, device='cuda')
        self.max_dual_value = -1e9
        for i in range(self.num_combs):
            active_idxs = self.combinations[:,i]
            A, b = self.sub_A_and_b(active_idxs)
            # solve the linear system
            this_alpha = A.inverse().mv(b)
            this_alpha = this_alpha[:-1]

            # check if valid solution 
            if (this_alpha >= 0).all():
                alpha = torch.zeros(self.n, device='cuda')
                alpha[active_idxs] = this_alpha
                this_dual_value = self.dual(alpha)

                if self.print:
                    print('A')
                    print(A)
                    print('b')
                    print(b)
                    print('solution to linear system')
                    print(this_alpha)
                    print('dual')
                    print(this_dual_value)

                if this_dual_value > self.max_dual_value:
                    self.max_dual_value = this_dual_value
                    self.best_alpha = alpha

        self.update_diagnostics()

    @torch.autograd.no_grad()
    def update_diagnostics(self):
        alpha = self.best_alpha
        self.step_size = alpha[0]
        self.step_0 = alpha[0]
        if len(alpha) > 1:
            self.step_size_unclipped = alpha[1]
            self.step_1 = alpha[1]
        if len(alpha) > 2:
            self.step_2 = alpha[2]
        if len(alpha) > 3:
            self.step_3 = alpha[3]
        if len(alpha) > 4:
            self.step_4 = alpha[4]

        if self.print:
            print('dual')
            print(self.max_dual_value)
            print('alpha')
            print(self.best_alpha)
            # print('alig')
            # print(self.alig_step)
            # print('segd')
            # print(self.segd_step)
            input('press any key')

    @torch.autograd.no_grad()
    def sub_A_and_b(self, chosen_idx):
        extra_0 = torch.tensor([1]).bool().cuda()
        idxs = torch.cat((chosen_idx,extra_0),0)
        A_rows = self.A[idxs, :]
        this_A = A_rows[:, idxs]
        this_b = self.b[idxs]
        return this_A, this_b

    @torch.autograd.no_grad()
    def dual(self, alpha):
        A = self.A[:-1,:-1]
        b = self.b[:-1]
        return - 0.5 * alpha.mul(A.mv(alpha)).sum() + b.mul(alpha).sum()

    @torch.autograd.no_grad()
    def update_parameters(self):
        # update parameters of model
        for group in self.param_groups:
            if group['eta'] > 0.0:
                for p in group['params']:
                    p.copy_(self.state[p]['w'][0])
                    p.data.add_(self.update(p, group))
                    # Nesterov momentum
                    if group["momentum"]:
                        self.apply_momentum(p, group)

    @torch.autograd.no_grad()
    def update(self, p, group):
        update = 0
        for i, grad in enumerate(self.state[p]['grads']):
            update += self.best_alpha[i] * grad
        return - group['eta'] * update

    @torch.autograd.no_grad()
    def apply_momentum_nesterov(self, p, group):
        buffer = self.state[p]['momentum_buffer']
        momentum = group["momentum"]
        buffer.mul_(momentum).add_(self.update(p, group))
        p.data.add_(momentum, buffer)



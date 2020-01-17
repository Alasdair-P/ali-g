import torch
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import required

class SBD(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, weight_decay=0, eps=1e-9, adjusted_momentum=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay)
        super(SBD, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn

        self.mode = 'alig'
        self.mode = 'segd'
        self.mode = 'segd3'

        self.n = 3
        self.zero_plane = True

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'] = []

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.apply_momentum = self.apply_momentum_nesterov

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def step(self, loss, x, y):
        # populate grads
        self.loss_w_t = float(loss())
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['w'] = p.data.detach().clone()
                self.state[p]['grads'] = []
                self.state[p]['grads'].append(p.grad.data.detach().clone())

        self.calc_extra_gradient(x, y)

        self.construct_A_and_b()

        self.generate_combs()

        self.loop_over_combs()

        self.update_parameters()

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def calc_extra_gradient(self, x, y):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w'] - group['eta'] * self.state[p]['grads'][0])
                # self.state[p]['w_e'] = p.data.detach().clone()

        with torch.enable_grad():
            self.model.zero_grad()
            loss, _ = self.obj(self.model(x), y, x)
            loss.backward()
            self.loss_w_e = float(loss)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'].append(p.grad.data.detach().clone())

    @torch.autograd.no_grad()
    def construct_A_and_b(self):
        """
        construst A matrix and b vector here the zero index is the standard gradient
        the one index is the extra gradient and the 2 index is the ground plane
        """
        self.A = torch.ones(self.n + 1, self.n + 1, device='cuda')
        self.b = torch.zeros(self.n + 1, device='cuda')
        self.A[0:-1,0:-1] = 0
        self.A[-1,-1] = 0
        self.b[-1]=1
        self.b[0] += self.loss_w_t
        self.b[1] += self.loss_w_e

        loop_range = self.n
        if self.zero_plane:
            loop_range = self.n - 1

        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                self.b[1] += - eta * self.state[p]['grads'][0].mul(self.state[p]['grads'][1]).sum()
                for i in range(loop_range):
                    for j in range(loop_range):
                        g_i = self.state[p]['grads'][i]
                        g_j = self.state[p]['grads'][j]
                        self.A[i, j] += eta * (g_i * g_j).sum()

        # print('A')
        # print(self.A)
        # print('b')
        # print(self.b)

    @torch.autograd.no_grad()
    def generate_combs(self):
        # we're creating the binary representation for all numbers from 0 to N-1
        N = 2**self.n
        a = np.arange(N, dtype=int).reshape(1,-1)
        l = int(np.log2(N))
        b = np.arange(l, dtype=int)[::-1,np.newaxis]
        self.combinations = torch.Tensor(np.array(a & 2**b > 0, dtype=int)).bool().cuda()
        self.combinations = self.combinations[:,1:]

        # print('self.combinations')
        # print(self.combinations)

        # add line adding columns of combs and removing columns thats sum to one
        self.num_combs = self.combinations.size()[1]

    @torch.autograd.no_grad()
    def loop_over_combs(self):
        self.best_alphas = torch.zeros(self.n, device='cuda')
        self.max_dual_value = -1e9
        for i in range(self.num_combs):
            active_idxs = self.combinations[:,i]
            A, b = self.create_sub_A_and_b(active_idxs)
            # solve the linear system
            this_alpha = A.inverse().mv(b)
            this_alpha = this_alpha[:-1]

            # print('A')
            # print(A)
            # print('b')
            # print(b)
            # print('solution to linear system')
            # print(this_alpha)

            # check if valid solution 
            if (this_alpha >= 0).all():
                alpha = torch.zeros(self.n, device='cuda')
                alpha[active_idxs] = this_alpha
                this_dual_value = self.dual(alpha)

                # print('dual')
                # print(this_dual_value)
                # print('alpha')
                # print(alpha)

                if this_dual_value > self.max_dual_value:
                    self.max_dual_value = this_dual_value
                    self.best_alpha = alpha
                    self.step_size = alpha[0]
                    self.step_size_unclipped  = alpha[1]

    @torch.autograd.no_grad()
    def create_sub_A_and_b(self, chosen_idx):
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
            momentum = group["momentum"]
            if group['eta'] > 0.0:
                for p in group['params']:
                    p.data.copy_(self.state[p]['w'])
                    p.data.add_(self.update(p, group))
                    # Nesterov momentum
                    if momentum:
                        self.apply_momentum(p, group, momentum)

    @torch.autograd.no_grad()
    def update(self, p, group):
        update = 0
        for i, grad in enumerate(self.state[p]['grads']):
            update += self.best_alpha[i] * grad
        return - group['eta'] * update

    @torch.autograd.no_grad()
    def apply_momentum_nesterov(self, p, group, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(self.update(p, group))
        p.data.add_(momentum, buffer)



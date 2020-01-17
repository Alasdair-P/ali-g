import torch
import math
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import required

class SBD(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, weight_decay=0, eps=1e-8, adjusted_momentum=False, betas=(0.0, 0.99), amsgrad=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay, betas=betas, amsgrad=amsgrad, eps=eps)
        super(SBD, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn

        self.print = True
        self.print = False

        self.A_functions = [self.sgd_gradient, self.segd_gradient, self.adam_gradient]
        self.b_functions = [self.sgd_b, self.segd_b, self.sgd_b]
        self.zero_plane = True
        self.eps = eps
        '''
        Notes:
        ~ segd requires sgd.
        ~ segd always second in list with sgd at first place.
        ~ zero plane is not included in self.A_functions, instead the self.zero_plane = True is set.
        ~ self.sgd_b should be used for any method where the gradient is assumed to caluclated at the current point.
        '''
        self.n = len(self.A_functions)
        if self.zero_plane:
            self.n += 1

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if group['momentum']:
                    state['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
                if self.adam_gradient in self.A_functions:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                if self.svrg_gradient in self.A_functions:
                    state['ref_point'] = p.data.clone().detach()
                    state['ref_grad'] = p.data.clone().detach()

        self.apply_momentum = self.apply_momentum_nesterov

        self.bn_stats = {}

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def step(self, loss, x, y):

        self.x, self.y, self.loss_w_t = x, y, float(loss())
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['w'] = p.data.detach().clone()
                self.state[p]['grads'] = []

        for func in self.A_functions:
            func()

        self.construct_A_and_b()

        self.generate_combs()

        self.loop_over_combs()

        self.update_parameters()

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def sgd_gradient(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'].append(p.grad.data.detach().clone())

    @torch.autograd.no_grad()
    def sgd_b(self):
        return self.loss_w_t

    @torch.autograd.no_grad()
    def segd_gradient(self):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w'] - group['eta'] * self.state[p]['grads'][0])
                # self.state[p]['w_e'] = p.data.detach().clone()

        self.save_bn_stats()
        with torch.enable_grad():
            self.model.zero_grad()
            loss, _ = self.obj(self.model(self.x), self.y, self.x)
            loss.backward()
            self.loss_w_e = float(loss)
        self.load_bn_stats()

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'].append(p.grad.data.detach().clone())

    @torch.autograd.no_grad()
    def save_bn_stats(self):
        if len(self.bn_stats) == 0:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.bn_stats[module] = {}
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                stats = self.bn_stats[module]
                stats['running_mean'] = module.running_mean.data.clone()
                stats['running_var'] = module.running_var.data.clone()
                stats['num_batches_tracked'] = module.num_batches_tracked.data.clone()

    @torch.autograd.no_grad()
    def load_bn_stats(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                stats = self.bn_stats[module]
                module.running_mean.data.copy_(stats['running_mean'])
                module.running_var.data.copy_(stats['running_var'])
                module.num_batches_tracked.data.copy_(stats['num_batches_tracked'])

    @torch.autograd.no_grad()
    def segd_b(self):
        return self.loss_w_e + self.A[0,1]

    @torch.autograd.no_grad()
    def svrg_gradient(self):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['ref_point'])
                # self.state[p]['w_e'] = p.data.detach().clone()

        with torch.enable_grad():
            self.model.zero_grad()
            loss, _ = self.obj(self.model(x), y, x)
            loss.backward()

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'].append(p.grad.data.detach().clone())

    @torch.autograd.no_grad()
    def adam_gradient(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                adam_grad = exp_avg / bias_correction1 / denom / 1
                self.state[p]['grads'].append(adam_grad)

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

        loop_range = self.n
        if self.zero_plane:
            loop_range = self.n - 1

        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                for i in range(loop_range):
                    for j in range(loop_range):
                        g_i = self.state[p]['grads'][i]
                        g_j = self.state[p]['grads'][j]
                        self.A[i, j] += eta * (g_i * g_j).sum()

        for i, func in enumerate(self.b_functions):
            self.b[i] = func()

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



import itertools
import time
try:
    import torch
    import concurrent.futures
except ImportError:
    raise ImportError("PyTorch is not installed!")

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is not installed!")

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])

class SBD(torch.optim.Optimizer):
    def __init__(self, params, eta=None, n=1, momentum=0, projection_fn=None, eps=1e-8, debug=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, eps=eps)
        super(SBD, self).__init__(params_list, defaults)

        self.projection = projection_fn
        self.print = debug
        # self.print = True
        self.N = n
        self.eps = eps
        self.eta = eta
        self.solve_forward = True

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device
        self.last_alpha = torch.zeros(self.N,device = self.device)
        self.last_loss = 1e12
        self.reset_bundle()

        self.step_time = 0
        self.solve_time = 0

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def print_times(self):
        print('fraction of time spent in solve method:',self.solve_time/self.step_time*100)

    @torch.autograd.no_grad()
    def reset_bundle(self):
        self.n = 1
        self.max_dual_value = 0.0
        self.best_alpha = torch.zeros(self.N,device = self.device)
        self.losses = []
        self.Create_Q_and_b()
        self.alig_step = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'] = []

    @torch.autograd.no_grad()
    def step(self, loss):
        tic = time.time()
        self.update_bundle(loss())
        self.Update_Q_and_b()
        if self.solve_forward or self.n == self.N:
            toc = time.time()
            if self.n == 2:
                self.alig_solve()
            else:
                self.solve_dual()
            self.solve_time += time.time() - toc
        else:
            self.sgd_solve()
        self.update_parameters()
        self.update_diagnostics()
        if self.n == self.N:
            if self.projection is not None:
                self.projection()
            self.reset_bundle()
        self.step_time += time.time() - tic

    @torch.autograd.no_grad()
    def update_bundle(self, loss):
        self.losses.append(float(loss))
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grads'].append(p.grad.data.detach().clone())
                if self.n == 1:
                    state['w_t'] = p.data.detach().clone()
        self.n += 1

    @torch.autograd.no_grad()
    def Update_Q_and_b(self):
        # i is the current index , n is current the number of LA in bundle 
        i = self.n-1
        for group in self.param_groups:
            for p in group['params']:
                g_i = self.state[p]['grads'][i-1]
                for j in range(1,self.n):
                    g_j = self.state[p]['grads'][j-1]
                    self.Q[i, j] += group['eta'] * (g_i * g_j).sum()
                    if not (i==j):
                        self.Q[j, i] = self.Q[i, j]

        self.b[i] += self.losses[i-1] + (self.Q[i,:-1] * self.best_alpha).sum()
        if self.print:
            print('i',i)
            print('Q')
            print(self.Q)
            print('b')
            print(self.b)

    @torch.autograd.no_grad()
    def solve_system(self, i):
        device = self.Q_.device
        idxs = torch.zeros(self.N+1, device=device)
        idxs[:self.n-1] = torch.tensor(i, device=device)
        idxs[self.n-1] = 1
        idxs[-1] = 1
        idxs = idxs.bool()
        Q_rows = self.Q_[idxs, :]
        A = Q_rows[:, idxs]
        b = self.b_[idxs].view(-1,1)
        # solve the linear system
        try:
            this_alpha, _ = torch.solve(b.view(-1,1), A)
        except:
            print('------------------- no solution found -----------------')
            return 0, torch.zeros(self.N, device=device)
        this_alpha = this_alpha.view(-1)
        this_alpha = this_alpha[:-1]
        # check if valid solution 
        if (this_alpha >= 0).all():
            alpha = torch.zeros(self.N, device=device)
            active_idxs = idxs[:-1]
            alpha[active_idxs] = this_alpha
            Q = self.Q_[:-1,:-1]
            b = self.b_[:-1]
            dual_val = - 0.5 * alpha.mul(Q.mv(alpha)).sum() + b.mul(alpha).sum()
        else:
            alpha = torch.zeros(self.N, device=device)
            dual_val = -1
        return dual_val, alpha

    @torch.autograd.no_grad()
    def solve_dual(self):
        self.Q_ = self.Q.clone().cpu()
        self.b_ = self.b.clone().cpu()
        # tic = time.time()
        iters = [i for i in itertools.product([1,0], repeat=self.n-1)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            a = executor.map(self.solve_system, iters)
        results = [x for x in a]
        dual_vals, alphas = zip(*results)
        idx, val = argmax(dual_vals)
        if val.cuda() > self.max_dual_value:
            self.max_dual_value = val
            self.last_alpha = self.best_alpha.clone()
            self.best_alpha.copy_(torch.Tensor(alphas[idx]))
        else:
            if self.print:
                print('using last alpha')
        # t = time.time() - tic
        # print('time taken {time:.3f}'.format(time=t))
        # input('press any key')

    @torch.autograd.no_grad()
    def alig_solve(self):
        self.alig_step = (self.b[1]/(self.Q[1,1]+self.eps))
        if self.Q[1,1] > self.eps:
            if self.alig_step <= 1:
                self.max_dual_value = 0.5 * self.alig_step * self.b[1]
                self.best_alpha[1] = self.alig_step
            else:
                self.max_dual_value = -0.5 * self.Q[1,1] + self.b[1]
                self.best_alpha[1] = 1

    @torch.autograd.no_grad()
    def sgd_solve(self):
        self.alig_step = (self.b[1]/(self.Q[1,1]+self.eps))
        self.max_dual_value = 0
        self.best_alpha[self.n-1] = 1

    @torch.autograd.no_grad()
    def Create_Q_and_b(self):
        self.Q = torch.ones(self.N + 1, self.N + 1, device=self.device)
        self.Q[0:-1,0:-1] = 0
        # self.Q += self.eps * torch.eye(self.N + 1, device=self.device)
        self.Q[-1,-1] = 0
        self.Q[0,0] = 0
        self.b = torch.zeros(self.N + 1, device=self.device)
        self.b[-1] = 1

    @torch.autograd.no_grad()
    def update_diagnostics(self):
        alpha = self.best_alpha
        self.step_size = max(min(float(self.alig_step),1),0)
        self.step_0 = alpha[0]
        if len(alpha) > 1:
            self.step_size_unclipped = self.alig_step
            self.step_1 = alpha[1]
        if len(alpha) > 2:
            self.step_2 = alpha[2]
        if len(alpha) > 3:
            self.step_3 = alpha[3]
        if len(alpha) > 4:
            self.step_4 = alpha[4]

        if self.print:
            print('------------------------')
            print('best dual')
            print(self.max_dual_value)
            print('best alpha')
            print(self.best_alpha)
            print('alig step')
            print(self.step_size)
            print('------------------------')
            input('press any key')

    @torch.autograd.no_grad()
    def sub_A_and_b(self, idxs):
        Q_rows = self.Q[idxs, :]
        this_Q= Q_rows[:, idxs]
        this_b = self.b[idxs]
        return this_Q, this_b

    @torch.autograd.no_grad()
    def dual(self, alpha):
        Q = self.Q[:-1,:-1]
        b = self.b[:-1]
        return - 0.5 * alpha.mul(Q.mv(alpha)).sum() + b.mul(alpha).sum()

    @torch.autograd.no_grad()
    def update_parameters(self):
        # update parameters of model
        for group in self.param_groups:
            if group['eta'] > 0.0:
                for p in group['params']:
                    A_alpha = self.update(p,group)
                    p.data.copy_(self.state[p]['w_t'] + A_alpha)
                    if self.n == self.N:
                        if group["momentum"]:
                            self.apply_momentum_nesterov(p, group, A_alpha)

    @torch.autograd.no_grad()
    def update(self, p, group):
        update = 0
        for i, grad in enumerate(self.state[p]['grads']):
            update += self.best_alpha[i+1] * grad
        return - group['eta'] * update

    @torch.autograd.no_grad()
    def apply_momentum_nesterov(self, p, group, A_alpha):
        buffer = self.state[p]['momentum_buffer']
        momentum = group["momentum"]
        buffer.mul_(momentum).add_(A_alpha)
        p.data.add_(momentum, buffer)

if __name__ == "__main__":
    opt = SBD([torch.randn(1,device='cuda'),torch.randn(1,device='cuda')], eta=1, n=4, momentum=0.9, projection_fn=None, eps=1e-8, debug=True)
    print('done')

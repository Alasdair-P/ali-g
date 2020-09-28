import itertools
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
        self.n = 1
        self.eps = eps
        self.eta = eta

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device
        self.Create_bundle()

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def step(self, loss):
        self.update_bundle(loss())
        self.Update_Q_and_b()
        self.solve_dual()
        self.update_parameters()
        self.update_diagnostics()
        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def Create_bundle(self):
        self.steps = 0
        self.last_loss = 1e12
        self.last_alpha = torch.zeros(self.N,device = self.device)
        self.best_alpha = torch.zeros(self.N,device = self.device) # done
        self.max_dual_value = 0
        self.losses = [0 for _ in range(self.N-1)]
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'] = [torch.zeros_like(p.data) for _ in range(self.N-1)]
        self.Q = torch.ones(self.N + 1, self.N + 1, device=self.device)
        self.Q[0:-1,0:-1] = 0
        self.Q[-1,-1] = 0
        self.Q[0,0] = 0
        self.b = torch.zeros(self.N + 1, device=self.device)
        self.b[-1] = 1

    @torch.autograd.no_grad()
    def update_bundle(self, loss):
        self.steps += 1
        self.losses.pop(0) # done
        self.losses.append(float(loss))
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grads'].pop(0) # done
                state['grads'].append(p.grad.data.detach().clone())
        self.calc_all = False
        if (self.best_alpha[1] > 0) or (self.steps < self.N-1):
            self.calc_all = True
            self.max_dual_value = 0
            self.best_alpha = torch.zeros(self.N,device = self.device) # done
        else:
            best_alpha = self.best_alpha.clone()  # done
            self.best_alpha[1:-1] = best_alpha[2:]  # done
            self.best_alpha[-1] = 0
        self.Q[1:-2,1:-2] = self.Q[2:-1,2:-1]
        self.Q[-2,1:-1] = 0
        self.Q[1:-1,-2] = 0
        b = self.b.clone()
        b[1:-2] = self.b[2:-1]
        b[-2] = 0
        self.b = b

    @torch.autograd.no_grad()
    def Update_Q_and_b(self):
        i = self.N-1
        for group in self.param_groups:
            for p in group['params']:
                g_i = self.state[p]['grads'][i-1]
                for j in range(1,self.N):
                    g_j = self.state[p]['grads'][j-1]
                    self.Q[i, j] += group['eta'] * (g_i * g_j).sum()
                    if not (i==j):
                        self.Q[j, i] = self.Q[i, j]

        self.b[i] += self.losses[i-1]
        self.alig_step = (self.b[i]/(self.Q[i,i]+self.eps))

        if self.print:
            print('i',i)
            print('losses')
            print('Q')
            print(self.Q)
            print('b')
            print(self.b)

    @torch.autograd.no_grad()
    def solve_system(self, i):
        device = self.Q_.device
        idxs = torch.ones(self.N+1, device=device)
        if len(i) == self.N:
            idxs[:self.N] = torch.tensor(i, device=device)
        else:
            idxs[:self.N-1] = torch.tensor(i, device=device)
        idxs = idxs.bool()
        Q_rows = self.Q_[idxs, :]
        A = Q_rows[:, idxs]
        b = self.b_[idxs].view(-1,1)
        # solve the linear system
        try:
            this_alpha, _ = torch.solve(b.view(-1,1), A)
        except:
            if self.print:
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
        if self.calc_all:
            iters = [i for i in itertools.product([1,0], repeat=self.N)]
        else:
            iters = [i for i in itertools.product([1,0], repeat=self.N-1)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            a = executor.map(self.solve_system, iters)
        results = [x for x in a]
        dual_vals, alphas = zip(*results)
        idx, val = argmax(dual_vals)
        if val > self.max_dual_value:
            self.max_dual_value = val
            self.last_alpha = self.best_alpha.clone()
            self.best_alpha.copy_(torch.Tensor(alphas[idx]))
        else:
            if self.print:
                print('using last alpha')

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
            print('calc all', self.calc_all)
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
                    # state['w_t'] = p.data.detach().clone()
                    A_alpha = self.update(p,group)
                    p.data.add_(A_alpha)
                    if group["momentum"]:
                        self.apply_momentum_nesterov(p, group, A_alpha)

        for j in range(1,self.N):
            self.b[j] -= (self.Q[j,:-1] * self.best_alpha).sum()
            if self.print:
                print('b',self.b)

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

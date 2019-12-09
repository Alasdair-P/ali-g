import torch
import torch.optim as optim
from torch.optim.optimizer import required

class ALIG_SEGD(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, weight_decay=0, eps=1e-6):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay)
        super(ALIG_SEGD, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.projection = projection_fn
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def calc_extra_gradient(self, x, y):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w_t'] - group['eta'] * self.state[p]['g_t'])
                self.state[p]['w_e'] = p.data.detach().clone()

        with torch.enable_grad():
            self.model.zero_grad()
            loss, _ = self.obj(self.model(x), y, x)
            loss.backward()
            self.loss_w_e = float(loss)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['g_e'] = p.grad.data.clone()

    @torch.autograd.no_grad()
    def compute_step_sizes(self, closure):
        # calculate step size
        loss_t = float(closure())
        loss_e = self.loss_w_e

        numerator = 0
        denominator = 0
        grad_sqrd_norm = 0

        for group in self.param_groups:
            eta = group['eta']
            if group['eta'] > 0.0:
                for p in group['params']:
                    g_t = self.state[p]['g_t']
                    g_e = self.state[p]['g_e']

                    # calculate u
                    grad_sqrd_norm += eta * (g_t).norm()**2
                    numerator += eta * (g_e).norm()**2 - 2 * eta * g_e.mul(g_t).sum()
                    denominator += eta * (g_t - g_e).norm()**2

        step_size_unclipped_alig = loss_t / (grad_sqrd_norm + self.eps)
        self.step_size_unclipped_alig = float(step_size_unclipped_alig)
        self.step_size_alig = float(step_size_unclipped_alig.clamp(min=0,max=1))

        step_size_unclipped = (numerator + loss_t - loss_e) / (denominator + self.eps)
        self.step_size_unclipped = float(step_size_unclipped)
        self.step_size = float(step_size_unclipped.clamp(min=0,max=1))

        for group in self.param_groups:
            group["step_size_segd"] = self.step_size
            group["step_size_alig"] = self.step_size_alig

    @torch.autograd.no_grad()
    def update_alig(self, group, p):
        return group['step_size_alig'] * self.state[p]['g_t']

    @torch.autograd.no_grad()
    def update_segd(self, group, p):
        step_size = group['step_size_segd']
        return step_size * self.state[p]['g_t'] + (1 - step_size) * self.state[p]['g_e']

    @torch.autograd.no_grad()
    def update_parameters(self, x, y):
        if abs(self.step_size - 1) < 1e-4 and abs(self.step_size_alig - 1) < 1e-4:
            self.sgd_step += 1
            for group in self.param_groups:
                self.step_size = group['step_size_alig']
                group['step_size'] = group['step_size_alig']
                self.step_size_unclipped = group['step_size_segd']
                for p in group['params']:
                    update = self.update_alig(group, p)
                    p.data.copy_(self.state[p]['w_t'] - group['eta'] * update)
                    self.apply_momentum(p, update, group['eta'], group['momentum'])
        else:
            self.compute_best_step(x, y)

    @torch.autograd.no_grad()
    def compute_best_step(self, x, y):
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(self.state[p]['w_t'] - group['eta'] * self.update_segd(group, p))

        loss_segd, _ = self.obj(self.model(x), y, x)

        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(self.state[p]['w_t'] - group['eta'] * self.update_alig(group, p))

        loss_alig, _ = self.obj(self.model(x), y, x)

        if loss_alig < loss_segd:
            self.alig_step += 1
            for group in self.param_groups:
                self.step_size = group['step_size_alig']
                group['step_size'] = group['step_size_alig']
                self.step_size_unclipped = group['step_size_segd']
                for p in group['params']:
                    update = self.update_alig(group, p)
                    self.apply_momentum(p, update, group['eta'], group['momentum'])
        else:
            self.segd_step += 1
            for group in self.param_groups:
                self.step_size = group['step_size_segd']
                group['step_size'] = group['step_size_segd']
                self.step_size_unclipped = group['step_size_alig']
                for p in group['params']:
                    update = self.update_segd(group, p)
                    p.data.copy_(self.state[p]['w_t'] - group['eta'] * update)
                    self.apply_momentum(p, update, group['eta'], group['momentum'])

    @torch.autograd.no_grad()
    def step(self, loss, x, y):

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['w_t'] = p.data.detach().clone()
                self.state[p]['g_t'] = p.grad.data.detach().clone()

        self.calc_extra_gradient(x, y)

        self.compute_step_sizes(loss)

        self.update_parameters(x, y)

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum(self, p, update, eta, momentum):
        if momentum:
            buffer = self.state[p]['momentum_buffer']
            buffer.mul_(momentum).add_(-eta, update)
            p.add_(momentum, buffer)


import torch
import torch.optim as optim
from torch.optim.optimizer import required

class SEGD(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, weight_decay=0, eps=1e-6, adjusted_momentum=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay)
        super(SEGD, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eta_2 = eta
        self.eps = eps
        self.update = self.segd_1

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.apply_momentum = self.apply_momentum_nesterov

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def calc_extra_gradient(self, x, y):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w_t'] - group['eta'] * self.state[p]['g_t'])
                # self.state[p]['w_e'] = p.data.detach().clone()

        with torch.enable_grad():
            self.model.zero_grad()
            loss, _ = self.obj(self.model(x), y, x)
            loss.backward()
            self.loss_w_e = float(loss)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['g_e'] = p.grad.data.clone()

    @torch.autograd.no_grad()
    def compute_step_size(self, closure):
        # calculate step size
        loss_t = float(closure())
        loss_e = self.loss_w_e

        g_t_norm = 0
        g_e_norm = 0
        g_t_minus_g_e_norm = 0

        for group in self.param_groups:
            eta = group['eta']
            if group['eta'] > 0.0:
                for p in group['params']:
                    g_t = self.state[p]['g_t']
                    g_e = self.state[p]['g_e']
                    g_t_norm += eta * (g_t).norm()**2
                    g_t_minus_g_e_norm += eta * (g_t - g_e).norm()**2
                    g_e_norm += eta * (g_e).norm()**2
        # SEGD
        segd_step_size_unclipped = (g_t_norm - loss_t + loss_e) / (g_t_minus_g_e_norm + self.eps) # calcuates v
        self.step_size_unclipped = float(segd_step_size_unclipped)
        self.step_size = float(segd_step_size_unclipped.clamp(min=0,max=1))
        g = self.step_size

        # ALIG
        alig_step_size_unclipped = loss_t / ((1 - g) * g_t_norm + g * g_e_norm + (g**2 - g) * g_t_minus_g_e_norm + self.eps)
        self.step_size_unclipped_alig = float(alig_step_size_unclipped)
        self.step_size_alig = float(alig_step_size_unclipped.clamp(min=0,max=1))

        for group in self.param_groups:
            group["step_size"] = self.step_size

    @torch.autograd.no_grad()
    def update_parameters(self):
        # update parameters of model
        for group in self.param_groups:
            momentum = group["momentum"]
            if group['eta'] > 0.0:
                for p in group['params']:
                    p.data.copy_(self.state[p]['w_t'])
                    p.data.add_(self.update(p, group))
                    # Nesterov momentum
                    if momentum:
                        self.apply_momentum(p, group, momentum)

    @torch.autograd.no_grad()
    def step(self, loss, x, y):

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['w_t'] = p.data.detach().clone()
                self.state[p]['g_t'] = p.grad.data.detach().clone()

        self.calc_extra_gradient(x, y)

        self.compute_step_size(loss)

        self.update_parameters()

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_nesterov(self, p, group, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(self.update(p, group))
        p.data.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def apply_momentum_polyak(self, p, group, momentum):
        p.data.add_(-self.w_update(p, group))
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(self.update(p, group))
        p.data.add_(buffer)

    @torch.autograd.no_grad()
    def segd_1(self, p, group):
        # update in terms of v
        alig_scaling = self.step_size_alig
        step_size = group['step_size']
        eta = group['eta'] * self.step_size_alig
        segd_update = - eta * step_size * self.state[p]['g_e'] - eta * (1 - step_size) * self.state[p]['g_t']
        return alig_scaling * segd_update

import torch
import torch.optim as optim
from torch.optim.optimizer import required

class SEGD3(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, weight_decay=0, eps=1e-9, adjusted_momentum=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, weight_decay=weight_decay)
        super(SEGD3, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eps = eps
        self.update = self.update_w
        self.step_0 = 0
        self.step_1 = 0
        self.step_2 = 0
        self.step_3 = 0

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.apply_momentum = self.apply_momentum_nesterov

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def dual(self, alpha_1, alpha_2):
        dual_value = - 0.5 * (alpha_1**2*self.eta_a_11 + 2*alpha_1*alpha_2*self.eta_a_12 + alpha_2**2*self.eta_a_22) + alpha_1 * self.b_1 + alpha_2 * self.b_2
        return dual_value

    @torch.autograd.no_grad()
    def alpha_1_eq_0(self):
        alpha_1 = 0
        alpha_2 = (self.b_2/(self.eta_a_22+self.eps)).clamp(min=0, max=1)
        return alpha_1, alpha_2

    @torch.autograd.no_grad()
    def alpha_2_eq_0(self):
        alpha_1 = (self.b_1/(self.eta_a_11+self.eps)).clamp(min=0, max=1)
        alpha_2 = 0
        return alpha_1, alpha_2

    @torch.autograd.no_grad()
    def alpha_3_eq_0(self):
        num = self.eta_a_22 - self.eta_a_12 + self.b_1 - self.b_2
        alpha_1 = (num/(self.d_1+self.eps)).clamp(min=0, max=1)
        alpha_2 = 1 - alpha_1
        return alpha_1, alpha_2

    @torch.autograd.no_grad()
    def alpha_123_neq_0(self):
        d_2 = self.eta_a_12*self.a_12 - self.eta_a_11 * self.a_22
        alpha_1 = (self.b_2*self.a_12 - self.b_1*self.a_22)/(d_2+self.eps)
        alpha_2 = (self.b_1*self.a_12 - self.b_2*self.a_11)/(d_2+self.eps)
        return alpha_1, alpha_2

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

        self.b_1 = loss_t
        self.b_2 = loss_e
        self.eta_a_11 = 0
        self.eta_a_12 = 0
        self.eta_a_22 = 0
        self.a_11 = 0
        self.a_12 = 0
        self.a_22 = 0
        self.d_1 = 0

        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                g_t = self.state[p]['g_t']
                g_e = self.state[p]['g_e']

                self.b_2 += eta * (g_e * g_t).sum()
                self.eta_a_11 += eta * (g_t * g_t).sum()
                self.eta_a_12 += eta * (g_t * g_e).sum()
                self.eta_a_22 += eta * (g_e * g_e).sum()
                self.a_11 += (g_t * g_t).sum()
                self.a_12 += (g_t * g_e).sum()
                self.a_22 += (g_e * g_e).sum()
                self.d_1 += eta * (g_t - g_e).norm()**2

        alpha_1, alpha_2 = self.alpha_123_neq_0()
        alpha_3 = 1 - alpha_1 - alpha_2
        if not (alpha_1 > 0 and alpha_2 > 0 and alpha_3 > 0):
            alp_11, alp_12 = self.alpha_1_eq_0()
            alp_21, alp_22 = self.alpha_2_eq_0()
            alp_31, alp_32 = self.alpha_3_eq_0()
            dual_vals = torch.tensor([self.dual(alp_11, alp_12), self.dual(alp_21, alp_22), self.dual(alp_31, alp_32)])
            # print('argmax:', dual_vals.argmax())
            if dual_vals.argmax() < 0.5:
                self.step_0 += 1
            elif dual_vals.argmax() < 1.5:
                self.step_1 += 1
            else:
                self.step_2 += 1
            my_list = [self.alpha_1_eq_0, self.alpha_2_eq_0, self.alpha_3_eq_0]
            alpha_1, alpha_2 = my_list[int(dual_vals.argmax())]()
        else:
            # print('argmax: 3')
            self.step_3 += 1

        self.step_size = alpha_1
        self.step_size_unclipped  = alpha_2

        for group in self.param_groups:
            group["alpha_1"] = alpha_1
            group["alpha_2"] = alpha_2

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
    def update_w(self, p, group):
        eta = group['eta']
        return - eta * group["alpha_1"] * self.state[p]['g_t'] - eta * group["alpha_2"] * self.state[p]['g_e']


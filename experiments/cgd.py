import torch
import torch.optim as optim

from torch.optim.optimizer import required


class CGD(optim.Optimizer):
    def __init__(self, params, model, obj, eta=None, momentum=0, projection_fn=None, debug=False, eps=1e-3, adjusted_momentum=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None)
        super(CGD, self).__init__(params_list, defaults)

        self.model = model
        self.obj = obj
        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.r = eps

        if debug == True:
            self.step_type = 'debug'
        else:
            self.step_type = 'fd'

        self.step_counter = 0
        self.first_order_steps = 0
        self.lower_bound_steps = 0
        self.fd_error = 0

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        if self.adjusted_momentum:
            self.apply_momentum = self.apply_momentum_adjusted
        else:
            self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()

    @torch.autograd.enable_grad()
    def exact_Hv_prod(self):
        w, g, v = [], [], []
        for p in self.model.parameters():
            w.append(p)
            g.append(p.grad)
            if p in self.state:
                v.append(self.state[p]['descent_dir'])
            else:
                v.append(torch.zeros_like(p,requires_grad=False))

        Hv = torch.autograd.grad(g, w, v)

        if 'debug' in self.step_type:
            for p, hv in list(zip(self.model.parameters(), Hv)):
                if p in self.state:
                    with torch.no_grad():
                        self.state[p]['Hv_exact'] = hv.clone()
        else:
            for p, hv in list(zip(self.model.parameters(), Hv)):
                if p in self.state:
                    with torch.no_grad():
                        self.state[p]['Hv'] = hv.clone()

    @torch.autograd.no_grad()
    def finite_difference_Hv_prod(self, x, y):

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w_0'] + self.r * self.state[p]['descent_dir'])

        with torch.enable_grad():
            self.model.zero_grad()
            loss = self.obj(self.model(x), y)
            loss.backward()

        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self.state[p]['w_0'] - self.r * self.state[p]['descent_dir'])
                self.state[p]['Hv'] = p.grad.data.clone()

        with torch.enable_grad():
            self.model.zero_grad()
            loss = self.obj(self.model(x), y)
            loss.backward()

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['Hv'] -= p.grad.data.clone()
                self.state[p]['Hv'] /= 2 * self.r
                p.copy_(self.state[p]['w_0'])

        # check finite difference error
        if 'debug' in self.step_type:
            self.fd_error = 0
            for n, p in self.model.named_parameters():
                if p in self.state:
                    hv_exact = self.state[p]['Hv_exact']
                    hv = self.state[p]['Hv']
                    self.fd_error += float((hv - hv_exact).abs().sum()/hv_exact.abs().sum()/len(group['params'])/len(self.param_groups))
            print(self.fd_error)

    @torch.autograd.no_grad()
    def compute_step_size(self, closure):
        # calculate step size
        loss = float(closure())
        h = 0
        h_eta = 0
        g = 0
        g_eta = 0
        for group in self.param_groups:
            eta = group['eta']
            if group['eta'] > 0.0:
                for p in group['params']:
                    d = self.state[p]['descent_dir']
                    hv = self.state[p]['Hv']
                    h += hv.mul(d).sum()
                    h_eta += hv.mul(d).sum()*eta
                    g += d.mul(d).sum()
                    g_eta += d.mul(d).sum()*eta

        max_value = float(( g - (g**2 - 2*loss*h).sqrt() )/h)
        self.step_size_unclipped = float(g_eta / (g + h_eta))

        if h < 0:
            self.first_order_steps += 1
            if (g +  h_eta) < 0:
                self.step_size = max_value
                self.lower_bound_steps += 1
            else:
                self.step_size = min(max_value, self.step_size_unclipped)
                if max_value < self.step_size_unclipped:
                    self.lower_bound_steps += 1
        else:
            # h > 0
            if (2*loss*h) > (g**2):
                self.step_size = self.step_size_unclipped
            else: #g**2 > 2*loss*h
                self.step_size = min(max_value, self.step_size_unclipped)
                if max_value < self.step_size_unclipped:
                    self.lower_bound_steps += 1

        for group in self.param_groups:
            group["step_size"] = self.step_size

    @torch.autograd.no_grad()
    def update_parameters(self):
        # update parameters of model
        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(-step_size, p.grad)
                # Nesterov momentum
                if momentum:
                    self.apply_momentum(p, step_size, momentum)

    @torch.autograd.no_grad()
    def step(self, loss, x, y):
        self.step_counter += 1
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['w_0'] = p.data.clone()
                self.state[p]['descent_dir'] = -p.grad.data.clone()

        # calculate hessien vector product
        if 'exact' in self.step_type or 'debug' in self.step_type:
            self.exact_Hv_prod()
        if 'fd' in self.step_type or 'debug' in self.step_type:
            self.finite_difference_Hv_prod(x, y)

        self.compute_step_size(loss)

        self.update_parameters()

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad)
        p.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def apply_momentum_adjusted(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).sub_(p.grad)
        p.add_(step_size * momentum, buffer)














































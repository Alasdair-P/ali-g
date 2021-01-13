try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("PyTorch is not installed, impossible to import `alig.th.AliG`")


class AliG2(torch.optim.Optimizer):
    r"""
    Implements the Adaptive Learning-rate for Interpolation with Gradients (ALI-G) algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        max_lr (float): maximal learning rate
        momentum (float, optional): momentum factor (default: 0)
        projection_fn (function, optional): projection function to enforce constraints (default: None)
        eps (float, optional): small constant for numerical stability (default: 1e-5)
        adjusted momentum (bool, optional): if True, use pytorch-like momentum, instead of standard Nesterov momentum

    Example:
        >>> optimizer = AliG(model.parameters(), max_lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fn(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step(lambda: float(loss_value))

    .. note::
        In order to compute the step-size, this optimizer requires a closure at every step
        that gives the current value of the loss function.
    """

    def __init__(self, params, max_lr=None, momentum=0, projection_fn=None, data_size=None, eps=1e-5, adjusted_momentum=False):
        if max_lr is not None and max_lr <= 0.0:
            raise ValueError("Invalid max_lr: {}".format(max_lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=max_lr, momentum=momentum, step_size=None)
        super(AliG2, self).__init__(params_list, defaults)

        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eps = eps
        self.sgd_mode = False
        self.first_update = True

        self.print = True
        self.print = False

        for group in self.param_groups:
            group["step_size"] = 0
            for p in group['params']:
                self.state[p]['mean'] = torch.zeros_like(p.data, requires_grad=False)
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device

        if data_size:
            # losses = np.load('losses.npy')
            # self.fhat = torch.Tensor(losses).to(self.device)
            # print('loaded losses, mean value:', self.fhat.mean(), self.fhat[:10])
            # input('press any key')
            self.fhat = torch.zeros(data_size, device=self.device)
            self.delta = torch.zeros(data_size, device=self.device)
            self.fxbar = torch.ones(data_size, device=self.device) * 1e6
            self.fx = torch.ones(data_size, device=self.device) * 1e6
            self.mean_grad = torch.zeros(1, device=self.device)
            self.step_0 = self.fhat.mean()
            self.step_1 = self.fxbar.mean()
            self.counter = 0

        self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def update_lb(self):
        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))

        reached_lb = (self.fxbar.le(self.fhat+self.delta*1e-1)).float()
        self.delta = (0.5*self.fxbar - 0.5*self.fhat).mul(1-reached_lb) + self.delta.mul(reached_lb)
        self.fhat = (0.5*self.fhat + 0.5*self.fxbar).mul(1-reached_lb) + (self.fhat - self.delta).clamp(min=0).mul(reached_lb)
        self.delta = self.delta.mul(1+reached_lb)
        if self.first_update:
            self.delta = (0.5*self.fxbar - 0.5*self.fhat)
            self.first_update = False

        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))
            print('delta', float(self.delta.mean()), self.delta)
            print('average fhat', float(self.fhat.mean()))
            input('press any key')

        if self.sgd_mode:
            for param_group in self.param_groups:
                param_group['max_lr'] *= 0.1
            self.step_size = self.param_groups[0]['max_lr']
            self.step_size_unclipped = self.param_groups[0]['max_lr']

        self.step_2 = reached_lb.mean()

    @torch.autograd.no_grad()
    def epoch(self):
        if self.fx.mean() < self.fxbar.mean():
            self.fxbar = self.fx
        if self.sgd_mode:
            print('savign to...', )
            np.save('losses_.npy', self.fxbar.detach().cpu().numpy())
        self.step_0 = self.fhat.mean()
        self.step_1 = self.fxbar.mean()
        self.step_4 = self.delta.mean()

        self.compute_step_size()
        self.counter = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mean'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.autograd.no_grad()
    def compute_step_size(self):

        # compute squared norm of gradient
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                grad_sqrd_norm += (self.state[p]['mean']/self.counter).norm() ** 2

        # self.step_size_unclipped = float((losses - lbs).mean() / (2 * grad_sqrd_norm + 1e-6))
        self.step_size_unclipped = float((self.fxbar - self.fhat).mean() / (2 * grad_sqrd_norm + 1e-6))

        if self.sgd_mode:
            for group in self.param_groups:
                self.step_size_unclipped = group["max_lr"]

        # if self.print:
        if True:
            print('fxbar', float(self.fxbar.mean()),'fhat', float(self.fhat.mean()), '|g|^2', float(grad_sqrd_norm), 'step_size: ', self.step_size_unclipped)
            print('n',self.counter)

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = min(self.step_size_unclipped, group["max_lr"])
            else:
                # print('max_lr is None')
                group["step_size"] = self.step_size_unclipped

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))

    @torch.autograd.no_grad()
    def step(self, closure):
        idx, losses = closure()
        self.counter += 1

        if self.step_size_unclipped is None:
            self.step_size_unclipped = 0.0
            self.step_size = 0.0

        self.fx[idx] = losses

        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['mean'] += p.grad.data
                p.add_(-step_size, p.grad)
                # Nesterov momentum
                if momentum:
                    self.apply_momentum(p, step_size, momentum)

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad)
        p.add_(momentum, buffer)

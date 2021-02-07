try:
    import torch
except ImportError:
    raise ImportError("PyTorch is not installed")


class PSDG(torch.optim.Optimizer):
    def __init__(self, params, max_lr=None, momentum=0, projection_fn=None):
        if max_lr is not None and max_lr <= 0.0:
            raise ValueError("Invalid max_lr: {}".format(max_lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=max_lr, momentum=momentum, step_size=None)
        super(AliG2, self).__init__(params_list, defaults)

        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.sgd_mode = True

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device

        self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def step(self, closure):

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

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad)
        p.add_(momentum, buffer)


if __name__ == "__main__":
    pass


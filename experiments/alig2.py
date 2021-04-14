try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("PyTorch is not installed, impossible to import `alig.th.AliG`")


class AliG2(torch.optim.Optimizer):
    r"""
       GLOBAL VERSION
    """

    def __init__(self, params, max_lr=None, momentum=0, projection_fn=None, data_size=None, transforms_size=0, path_to_losses=None, global_lb=True, save=False, eps=1e-5,adjusted_momentum=False):
        # if max_lr is not None and max_lr <= 0.0:
            # raise ValueError("Invalid max_lr: {}".format(max_lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=max_lr, momentum=momentum, step_size=None)
        super(AliG2, self).__init__(params_list, defaults)

        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eps = eps
        self.sgd_mode = save
        # self.sgd_mode = True
        self.first_update = True
        self.path_to_losses = path_to_losses
        self.eta_is_zero = (max_lr == 0.0)
        self.global_lb = global_lb

        self.print = True
        self.print = False

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device

        if data_size:
            if self.path_to_losses:
                losses = np.load(path_to_losses)
                self.fhat = torch.Tensor(losses).to(self.device) # Estimated optimal value
                # self.fhat = torch.zeros(data_size, device=self.device).fill_(float(path_to_losses)) # Estimated optimal value
            else:
                self.fhat = torch.zeros(data_size, device=self.device) # Estimated optimal value
            print('mean EOV: {m_loss}'.format(m_loss=self.fhat.mean()))
            self.delta = torch.zeros(data_size, device=self.device)
            self.fxbar = torch.ones(data_size, device=self.device) * 1e6 # Loss of sample from best epoch
            self.fx = torch.ones(data_size, device=self.device) * 1e6 # Current loss of samble this epoch
            self.step_0 = self.fhat.mean()
            self.step_1 = self.fxbar.mean()
            self.step_2 = self.fx.mean()

            # self.fhat = torch.zeros(1, device=self.device).fill_(2.2) # Estimated optimal value
            # self.fxbar = torch.ones(1, device=self.device) * 1e6 # Loss of sample from best epoch
            # self.delta = torch.zeros(1, device=self.device)

        self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()


    @torch.autograd.no_grad()
    def update_lb(self):
        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))

        # reached_lb = (self.fxbar.le(self.fhat+self.delta*0.1)).float()
        reached_lb = (self.fxbar.le(self.fhat+self.delta*1.0)).float()
        self.delta = (0.5*self.fxbar - 0.5*self.fhat).mul(1-reached_lb) + self.delta.mul(reached_lb)
        self.fhat = (0.5*self.fhat + 0.5*self.fxbar).mul(1-reached_lb) + (self.fhat - 0.5 * self.delta).clamp(min=0).mul(reached_lb)
        #print('UPDATE MEAN')
        #self.fhat = (0.5*self.fhat + 0.5*self.fxbar)
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
            if self.global_lb:
                self.fxbar.fill_(self.fx.mean())
            else:
                self.fxbar = self.fx
        self.step_1 = self.fxbar.mean()
        if self.sgd_mode:
            print('savign losses')
            # np.save('model_dataset.npy', self.fxbar.detach().cpu().numpy())
            np.save('losses.npy', self.fxbar.detach().cpu().numpy())
        self.step_0 = self.fhat.mean()
        self.step_4 = self.delta.mean()
        self.step_2 = self.fx.mean()

    @torch.autograd.no_grad()
    def compute_step_size(self, losses, lbs):

        # compute squared norm of gradient
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    continue
                grad_sqrd_norm += p.grad.data.norm() ** 2

        # compute unclipped step-size
        # self.step_size_unclipped = float((losses - lbs).mean() / (2 * grad_sqrd_norm + 1e-6))
        self.step_size_unclipped = float((losses - lbs).mean() / (grad_sqrd_norm + 1e-4))
        self.step_3 = grad_sqrd_norm
        self.step_5 = losses.mean()

        if self.sgd_mode:
            for group in self.param_groups:
                self.step_size_unclipped = group["max_lr"]

        if self.print:
            print('losses', losses, 'lbs', lbs, 'numerator', float((losses - lbs).mean()), '|g|^2', float(grad_sqrd_norm), 'step_size: ', self.step_size_unclipped)
            input('press any key')

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = max(min(self.step_size_unclipped, group["max_lr"]),0.0)
            else:
                # print('max_lr is None')
                group["step_size"] = self.step_size_unclipped

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))

    @torch.autograd.no_grad()
    def step(self, closure):
        idx, losses = closure()
        self.fx[idx] = losses

        # losses = losses.clamp(min=float(self.fhat))
        # losses = losses.mean()
        # losses = losses.clamp(min=float(self.fhat))



        if self.eta_is_zero:
            return

        fhat = self.fhat[idx]
        if self.print:
            print('idx', idx, 'losses', losses, 'fbar', fhat)
        self.compute_step_size(losses, fhat)


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
    opt = AliG2(torch.Tensor([10]), max_lr=None, momentum=0, projection_fn=None, data_size=1)


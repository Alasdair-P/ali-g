import torch
import math
import numpy as np
import torch.nn as nn

class Reg(object):

    def __init__(self, args, model):
        print('w - reg')
        self.temperature = 0.0
        self.beta_scale = args.beta_scale
        self.beta_rate = args.beta_rate
        self.t = 0
        self.epoch = 0
        self.model = model
        self.active = args.reg
        self.percent_binary = 0.0
        self.average_distance = 0.0
        self.normalised_weight_norm = 0.0
        self.xp_name = args.xp_name
        self.hq_epoch = args.hq_epoch

    def if_binary(self, n):
        return (not('bn' in n) and not('downsample' in n)
            and not('fc' in n and 'bias' in n))

    def iter_update(self):
        if self.active:
            self.w_regularization_step()
            self.t += 1

    def epoch_update(self):
        self.calc_dist_to_binary()
        self.epoch += 1
        if self.active:
            self.calc_new_beta()

    def calc_new_beta(self):
        self.temperature = self.epoch * self.beta_scale

    def calc_dist_to_binary(self):
        marg = 10**-4
        num_w_non_binary = 0.0
        num_weights = 0.0
        distance = 0.0
        w_norm = 0.0
        max_w = -1
        min_w = 1
        for n, p in self.model.named_parameters():
            p = p.data
            if self.if_binary(n):
                w = p.data
                num_weights += w.numel()
                distance += (w.sign() - w).abs().sum()
                max_w = max(max_w, w.max())
                min_w = min(min_w, w.min())
                num_w_non_binary += ((w < -1-marg) + (w > -1+marg) * (w < 1-marg) + (w > 1+marg)).sum()
            w_norm += p.data.norm()**2
        if num_weights > 0:
            self.average_distance = float(distance/num_weights)
            self.normalised_weight_norm = float(w_norm.sqrt())
            self.percent_binary = float(num_w_non_binary)*100/float(num_weights)
            print('Average Distance: ',  "{0:.6f}".format(self.average_distance))
            print('Percentage non-binary: ', "{0:.6f}".format(self.percent_binary))
            print('Max w: ', "{0:.6f}".format(float(max_w)))
            print('Min w: ', "{0:.6f}".format(float(min_w)))
            print('beta/temp: ', self.temperature)
            print('t', self.t)
        else:
            print('no layers of correct type found!')

    # def soft_threshold(self, w):
        # return w * (1 + self.temperature)

    def soft_threshold(self, w):
        return w.sign() + (w - w.sign()).sign() * ((w - w.sign()).abs() - self.temperature).clamp(min=0)

    # def soft_threshold(self, w):
        # return (((w * w.sign()) + self.temperature).clamp(max=1)) * w.sign()

    def w_regularization_step(self):
        if self.active:
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    if self.temperature > 0.0:
                        p.data = self.soft_threshold(p.data)
                        p.data.clamp_(-1,1)

    def hard_quantize(self, optimizer):
        optimizer.param_groups[1]['eta'] = 0
        optimizer.param_groups[1]['lr'] = 0
        self.calc_dist_to_binary()
        print('------------- hard quantize ------------')
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                p.data.copy_(p.data.sign())
                p.requires_grad = False
        self.calc_dist_to_binary()




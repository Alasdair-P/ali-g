import torch
import torch.nn as nn

def get_loss(args):
    if args.opt == 'dfw' or args.loss == 'svm':
        loss_fn = MultiClassHingeLoss()
        if 'cifar' in args.dataset:
            args.smooth_svm = True
    elif args.loss == 'map':
        loss_fn = ap(n_classes=args.n_classes)
    elif args.loss == 'ndcg':
        loss_fn = NDCG(n_classes=args.n_classes)
    elif args.dataset == 'imagenet':
        return EntrLoss(n_classes=args.n_classes-1)
    elif args.loss == 'norm_ce':
        loss_fn = NormCE()
    else:
        loss_fn = nn.CrossEntropyLoss()

    print('L2 regularization: \t {}'.format(args.weight_decay))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn

class MultiClassHingeLoss(nn.Module):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and
    output `y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)`):

    This implements a Crammer & Singer formulation, which penalizes the maximal margin violation.
    Note that `torch.nn.MultiMarginLoss` uses the Weston & Watkins formulation,
    which penalizes the sum of the margin violations and performs significantly worse in our experience.
    """

    smooth = False

    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()
        self.smooth = False
        self._range = None

    def forward(self, x, y):
        aug = self._augmented_scores(x, y)
        xi = self._compute_xi(x, aug, y)

        loss = torch.sum(aug * xi) / x.size(0)
        return loss

    def _augmented_scores(self, s, y):
        if self._range is None:
            delattr(self, '_range')
            self.register_buffer('_range', torch.arange(s.size(1), device=s.device)[None, :])

        delta = torch.ne(y[:, None], self._range).detach().float()
        return s + delta - s.gather(1, y[:, None])

    @torch.autograd.no_grad()
    def _compute_xi(self, s, aug, y):

        # find argmax of augmented scores
        _, y_star = torch.max(aug, 1)
        # xi_max: one-hot encoding of maximal indices
        xi_max = torch.eq(y_star[:, None], self._range).float()

        if MultiClassHingeLoss.smooth:
            # find smooth argmax of scores
            xi_smooth = nn.functional.softmax(s, dim=1)
            # compute for each sample whether it has a positive contribution to the loss
            losses = torch.sum(xi_smooth * aug, 1)
            mask_smooth = torch.ge(losses, 0).float()[:, None]
            # keep only smoothing for positive contributions
            xi = mask_smooth * xi_smooth + (1 - mask_smooth) * xi_max
        else:
            xi = xi_max

        return xi

    def __repr__(self):
        return 'MultiClassHingeLoss()'

class set_smoothing_enabled(object):
    def __init__(self, mode):
        """
        Context-manager similar to the torch.set_grad_enabled(mode).
        Within the scope of the manager the MultiClassHingeLoss is smoothed (it is not otherwise).
        """
        MultiClassHingeLoss.smooth = bool(mode)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiClassHingeLoss.smooth = False

class EntrLoss(nn.Module):
    """Implementation from https://github.com/locuslab/lml/blob/master/smooth-topk/src/losses/entr.py.

    The MIT License

    Copyright 2019 Intel AI, CMU, Bosch AI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    def __init__(self, n_classes, k=5, tau=1.0):
        super(EntrLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

    def forward(self, x, y):
        n_batch = x.shape[0]

        x = x/self.tau
        x_sorted, I = x.sort(dim=1, descending=True)
        x_sorted_last = x_sorted[:,self.k:]
        I_last = I[:,self.k:]

        fy = x.gather(1, y.unsqueeze(1))
        J = (I_last != y.unsqueeze(1)).type_as(x)

        # Could potentially be improved numerically by using
        # \log\sum\exp{x_} = c + \log\sum\exp{x_-c}
        safe_z = torch.clamp(x_sorted_last-fy, max=80)
        losses = torch.log(1.+torch.sum(safe_z.exp()*J, dim=1))

        return losses.mean()

class NormCE(nn.Module):

    def __init__(self):
        super(NormCE, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, scores, y):
        scores = (scores-scores.mean(dim=1,keepdim=True)).div(scores.std(dim=1, keepdim=True))
        return self.loss(scores, y)

class AP(nn.Module):
    """Implementation of <Efficient Optimization for Rank-based Loss Functions>.
    """

    def __init__(self, n_classes):
        super(AP, self).__init__()
        self.print = True
        self.print = False
        self.count = 0
        self.n_classes = n_classes

    def calc_optimal_interleaving_rank(self, l_minus, r_minus, l_plus, r_plus):
        self.count += 1
        l_minus = int(l_minus)
        l_plus = int(l_plus)
        r_minus = int(r_minus)
        r_plus = int(r_plus)

        if self.print:
            print('l_minus', l_minus,'r_minus', r_minus, 'l_plus', l_plus, 'r_plus', r_plus)
            print('opt', self.opt)

        if l_plus == r_plus:
            self.opt[l_minus:r_minus] = l_plus
            # print('opt', self.opt)
            return

        m = self.median(l_minus, r_minus)
        m = self.select(m, l_minus, r_minus)
        opt_m = self.opt_j(l_plus, r_plus, m)
        self.opt[m].copy_(opt_m)
        if l_minus < m:
            self.calc_optimal_interleaving_rank(l_minus, m, l_plus, opt_m)
        if m+1 < r_minus:
            self.calc_optimal_interleaving_rank(m+1, r_minus, opt_m, r_plus)

    @torch.autograd.no_grad()
    def opt_j(self, l_plus, r_plus, m):
        # range of i
        j = m + 1
        k = torch.arange(r_plus, l_plus, device=self.s_plus.device)
        l = torch.arange(r_plus, l_plus+1, device=self.s_plus.device)
        # array where each element is single term in sum
        deltas = (j/(j+k.float())-(j-1)/(j+k.float()))/(self.P) - 2*(self.s_plus[(k-1)]-self.s_minus[m])/(self.P*self.N)
        deltas = torch.cat((deltas,torch.zeros(1, device=self.s_plus.device)),0)
        opt_vals = torch.cumsum(deltas.flip(0),0)
        # computes array were nth value is sum form n to r_plus
        idx = torch.argmax(opt_vals.flip(0))
        # compute argmax
        if self.print:
            print('r_plus', r_plus, 'l_plus', l_plus, 'm', m, 'deltas', deltas, 'opt_vals', opt_vals, 'idx', idx, 'opt', l[idx], 'k', l)
        return l[idx]

    @torch.autograd.no_grad()
    def median(self, l_minus, r_minus):
        _, m = torch.median(self.s_minus[l_minus:r_minus],0)
        return m

    def select(self, m, l_minus, r_minus):
        a = self.s_minus[l_minus: r_minus]
        less_than_m = a[(a<a[m])].view(-1)
        more_than_m = a[(a>a[m])].view(-1)
        equal_m = a[(a==a[m])].view(-1)
        self.s_minus[l_minus: r_minus] = torch.cat((less_than_m,equal_m,more_than_m), 0)
        return len(less_than_m)+l_minus

    @torch.autograd.no_grad()
    def calc_cs(self):
        r_plus = torch.zeros_like(self.s_plus)
        sum_ind_plus_over_ind = 0
        for i in range(self.P):
            r_plus[i] = 1 + (self.opt <= (i+1)).long().sum()
            sum_ind_plus_over_ind += (i+1)/(i + r_plus[i])
        Delta_R_R_star = 1 - (sum_ind_plus_over_ind/self.P)
        c_plus = self.N + 2 - 2 * r_plus
        c_minus = self.P + 2 - 2 * self.opt
        c_plus_star = self.N + 2 - 2 * torch.ones_like(self.s_plus)
        c_minus_star = self.P + 2 - 2 * torch.ones_like(self.s_minus) * (self.P+1)
        return c_plus, c_minus, c_plus_star, c_minus_star, Delta_R_R_star

    def calc_loss_for_cth_class(self, scores_2d, class_lables, this_class):
        self.count = 0
        scores = scores_2d.view(-1).float()

        with torch.no_grad():
            true_ranking = (class_lables.eq(this_class)).bool()
            self.C = true_ranking.shape[0]
            self.P = true_ranking.sum()
            self.N = self.C - self.P

        self.s_plus = scores[true_ranking]

        if (self.P == 0) or (self.N == 0):
            loss = self.s_plus.sum().mul(0)
            return loss

        with torch.no_grad():
            _, idx = self.s_plus.sort(descending=True)

        self.s_plus = self.s_plus[idx].float()
        self.s_minus = scores[true_ranking.logical_not()]
        self.opt = torch.zeros_like(self.s_minus)

        self.calc_optimal_interleaving_rank(0, self.N, self.P+1, 1)


        c_plus, c_minus, c_plus_star, c_minus_star, Delta_R_R_star = self.calc_cs()
        sc_plus = self.s_plus.mul(c_plus.detach()).sum()
        sc_minus = self.s_minus.mul(c_minus.detach()).sum()
        F_R = sc_plus + sc_minus
        sc_plus_star = self.s_plus.mul(c_plus_star.detach()).sum()
        sc_minus_star = self.s_minus.mul(c_minus_star.detach()).sum()
        F_R_star = sc_plus_star + sc_minus_star

        loss =  Delta_R_R_star + (F_R - F_R_star)/(self.P * self.N)

        # if loss < 0:
        if self.print:
            print('------------------------------------------')
            print('scores',scores)
            print('true_ranking', true_ranking)
            print('self.C',self.C)
            print('self.P',self.P)
            print('self.N',self.N)
            print('P*N',self.N*self.P)
            print('self.s_plus',self.s_plus)
            print('self.s_minus',self.s_minus)
            print('self.opt',self.opt)
            print('s_plus', self.s_plus, 's_minus', self.s_minus)
            print('c_plus, c_minus, c_plus_star, c_minus_star', c_plus, c_minus, c_plus_star, c_minus_star)
            print('sc_plus, sc_minus, sc_plus_star, sc_minus_star', sc_plus, sc_minus, sc_plus_star, sc_minus_star)
            print('F_R', F_R, 'F_R_star', F_R_star)
            print('loss', loss)
            print('Delta_R_R_star', Delta_R_R_star)
            print('------------------------------------------')
            input('press any key')

        return loss

    def forward(self, scores, class_lables):
        # scores = (scores-scores.mean(dim=1,keepdim=True)).div(scores.std(dim=1, keepdim=True))
        loss = 0
        for i in range(self.n_classes):
            loss += self.calc_loss_for_cth_class(scores[:,i], class_lables, i)
        return loss/self.n_classes

class ap(nn.Module):
    """Implementation of <Efficient Optimization for Rank-based Loss Functions>.
    """

    def __init__(self, n_classes):
        super(ap, self).__init__()
        self.print = True
        self.print = False
        self.count = 0
        self.n_classes = n_classes

    def calc_optimal_interleaving_rank(self, l_minus, r_minus, l_plus, r_plus):
        self.count += 1
        l_minus = int(l_minus)
        l_plus = int(l_plus)
        r_minus = int(r_minus)
        r_plus = int(r_plus)

        if self.print:
            print('l_minus', l_minus,'r_minus', r_minus, 'l_plus', l_plus, 'r_plus', r_plus)
            print('opt', self.opt)

        if l_plus == r_plus:
            self.opt[l_minus:r_minus] = l_plus
            # print('opt', self.opt)
            return

        m = self.median(l_minus, r_minus)
        m = self.select(m, l_minus, r_minus)
        opt_m = self.opt_j(l_plus, r_plus, m)
        self.opt[m].copy_(opt_m)
        if l_minus < m:
            self.calc_optimal_interleaving_rank(l_minus, m, l_plus, opt_m)
        if m+1 < r_minus:
            self.calc_optimal_interleaving_rank(m+1, r_minus, opt_m, r_plus)

    @torch.autograd.no_grad()
    def opt_j(self, l_plus, r_plus, m):
        # range of i
        j = m + 1
        k = torch.arange(r_plus, l_plus, device=self.s_plus.device)
        l = torch.arange(r_plus, l_plus+1, device=self.s_plus.device)
        # array where each element is single term in sum
        deltas = (j/(j+k.float())-(j-1)/(j+k.float()))/(self.P) - 2*(self.s_plus[(k-1)]-self.s_minus[m])/(self.P*self.N)
        deltas = torch.cat((deltas,torch.zeros(1, device=self.s_plus.device)),0)
        opt_vals = torch.cumsum(deltas.flip(0),0)
        # computes array were nth value is sum form n to r_plus
        idx = torch.argmax(opt_vals.flip(0))
        # compute argmax
        if self.print:
            print('r_plus', r_plus, 'l_plus', l_plus, 'm', m, 'deltas', deltas, 'opt_vals', opt_vals, 'idx', idx, 'opt', l[idx], 'k', l)
        return l[idx]

    @torch.autograd.no_grad()
    def median(self, l_minus, r_minus):
        _, m = torch.median(self.s_minus[l_minus:r_minus],0)
        return m

    def select(self, m, l_minus, r_minus):
        a = self.s_minus[l_minus: r_minus]
        less_than_m = a[(a<a[m])].view(-1)
        more_than_m = a[(a>a[m])].view(-1)
        equal_m = a[(a==a[m])].view(-1)
        self.s_minus[l_minus: r_minus] = torch.cat((less_than_m,equal_m,more_than_m), 0)
        return len(less_than_m)+l_minus

    @torch.autograd.no_grad()
    def calc_cs(self):
        r_plus = torch.zeros_like(self.s_plus)
        sum_ind_plus_over_ind = 0
        for i in range(self.P):
            r_plus[i] = 1 + (self.opt <= (i+1)).long().sum()
            sum_ind_plus_over_ind += (i+1)/(i + r_plus[i])
        Delta_R_R_star = 1 - (sum_ind_plus_over_ind/self.P)
        return Delta_R_R_star

    def calc_loss_for_cth_class(self, scores_2d, class_lables, this_class):
        self.count = 0
        scores = scores_2d.view(-1).float()

        with torch.no_grad():
            true_ranking = (class_lables.eq(this_class)).bool()
            self.C = true_ranking.shape[0]
            self.P = true_ranking.sum()
            self.N = self.C - self.P

        self.s_plus = scores[true_ranking]

        if (self.P == 0) or (self.N == 0):
            loss = self.s_plus.sum().mul(0)
            return loss

        with torch.no_grad():
            _, idx = self.s_plus.sort(descending=True)

        self.s_plus = self.s_plus[idx].float()
        self.s_minus = scores[true_ranking.logical_not()]
        self.opt = torch.zeros_like(self.s_minus)
        self.calc_optimal_interleaving_rank(0, self.N, self.P+1, 1)
        Delta_R_R_star = self.calc_cs()
        loss =  1 - Delta_R_R_star
        return loss

    def forward(self, scores, class_lables):
        # scores = (scores-scores.mean(dim=1,keepdim=True)).div(scores.std(dim=1, keepdim=True))
        loss = 0
        for i in range(self.n_classes):
            loss += self.calc_loss_for_cth_class(scores[:,i], class_lables, i)
        return loss/self.n_classes

class NDCG(nn.Module):
    """Implementation of <Efficient Optimization for Rank-based Loss Functions>.
    """

    def __init__(self, n_classes):
        super(NDCG, self).__init__()
        self.print = True
        self.print = False
        self.count = 0
        self.n_classes = n_classes

    def calc_optimal_interleaving_rank(self, l_minus, r_minus, l_plus, r_plus):
        self.count += 1
        l_minus = int(l_minus)
        l_plus = int(l_plus)
        r_minus = int(r_minus)
        r_plus = int(r_plus)

        if self.print:
            print('l_minus', l_minus,'r_minus', r_minus, 'l_plus', l_plus, 'r_plus', r_plus)
            print('opt', self.opt)


        if l_plus == r_plus:
            self.opt[l_minus:r_minus] = l_plus
            # print('opt', self.opt)
            return

        m = self.median(l_minus, r_minus)
        m = self.select(m, l_minus, r_minus)
        opt_m = self.opt_j(l_plus, r_plus, m)
        self.opt[m].copy_(opt_m)
        if l_minus < m:
            self.calc_optimal_interleaving_rank(l_minus, m, l_plus, opt_m)
        if m+1 < r_minus:
            self.calc_optimal_interleaving_rank(m+1, r_minus, opt_m, r_plus)

    """
    # NDCG
    @torch.autograd.no_grad()
    def opt_j(self, l_plus, r_plus, m):
        # range of i
        j = m + 1
        l = torch.arange(r_plus, l_plus+1, device=self.s_plus.device)
        deltas = (self.D(l+j-1)-self.D(self.P+j))/self.C - 2*(self.s_plus[(k-1)]-self.s_minus[m])/(self.P*self.N)
        idx = torch.argmax(deltas)
        if self.print:
            print('r_plus', r_plus, 'l_plus', l_plus, 'm', m, 'deltas', deltas,'idx', idx, 'opt', l[idx], 'l', l)
        return l[idx]
    """

    @torch.autograd.no_grad()
    def opt_j(self, l_plus, r_plus, m):
        # range of i
        j = m + 1
        k = torch.arange(r_plus, l_plus, device=self.s_plus.device)
        l = torch.arange(r_plus, l_plus+1, device=self.s_plus.device)
        # array where each element is single term in sum
        f =  - 2*(self.s_plus[(k-1)]-self.s_minus[m])/(self.P*self.N)
        f = torch.cat((f,torch.zeros(1, device=self.s_plus.device)),0)
        opt_vals = torch.cumsum(f.flip(0),0).flip(0)
        opt_vals += (self.D(l+j-1)-self.D(self.P+j))/self.C
        # computes array were nth value is sum form n to r_plus
        idx = torch.argmax(opt_vals)
        # compute argmax
        if self.print:
            print('r_plus', r_plus, 'l_plus', l_plus, 'm', m, 'f', f, 'opt_vals', opt_vals, 'idx', idx, 'opt', l[idx], 'k', l)
        return l[idx]

    @torch.autograd.no_grad()
    def median(self, l_minus, r_minus):
        _, m = torch.median(self.s_minus[l_minus:r_minus],0)
        return m

    def select(self, m, l_minus, r_minus):
        a = self.s_minus[l_minus: r_minus]
        less_than_m = a[(a<a[m])].view(-1)
        more_than_m = a[(a>a[m])].view(-1)
        equal_m = a[(a==a[m])].view(-1)
        self.s_minus[l_minus: r_minus] = torch.cat((less_than_m,equal_m,more_than_m), 0)
        return len(less_than_m)+l_minus

    @torch.autograd.no_grad()
    def D(self, i):
        return 1/torch.log2(1+i.float())

    # NDCG
    @torch.autograd.no_grad()
    def calc_cs(self):
        r_plus = torch.zeros_like(self.s_plus)
        num = torch.zeros_like(self.s_plus)
        for i in range(self.P):
            r_plus[i] = 1 + (self.opt <= (i+1)).long().sum()
            num[i] = self.D(i + r_plus[i])
        Delta_R_R_star = 1 - (num.sum())/self.C
        c_plus = self.N + 2 - 2 * r_plus
        c_minus = self.P + 2 - 2 * self.opt
        c_plus_star = self.N + 2 - 2 * torch.ones_like(self.s_plus)
        c_minus_star = self.P + 2 - 2 * torch.ones_like(self.s_minus) * (self.P+1)
        return c_plus, c_minus, c_plus_star, c_minus_star, Delta_R_R_star

    def calc_loss_for_cth_class(self, scores_2d, class_lables, this_class):
        self.count = 0
        scores = scores_2d.view(-1).float()

        with torch.no_grad():
            true_ranking = (class_lables.eq(this_class)).bool()
            self.C = true_ranking.shape[0]
            self.P = true_ranking.sum()
            self.N = self.C - self.P
            self.ind_plus = (torch.arange(scores.size(0), device=scores_2d.device)+1)[true_ranking]
            self.C = self.D(torch.arange(self.P, device=scores_2d.device)+1).sum()

        self.s_plus = scores[true_ranking]


        if (self.P == 0) or (self.N == 0):
            loss = self.s_plus.sum().mul(0)
            return loss

        with torch.no_grad():
            _, idx = self.s_plus.sort(descending=True)

        self.s_plus = self.s_plus[idx].float()
        self.s_minus = scores[true_ranking.logical_not()]
        self.opt = torch.zeros_like(self.s_minus)

        self.calc_optimal_interleaving_rank(0, self.N, self.P+1, 1)


        c_plus, c_minus, c_plus_star, c_minus_star, Delta_R_R_star = self.calc_cs()
        sc_plus = self.s_plus.mul(c_plus.detach()).sum()
        sc_minus = self.s_minus.mul(c_minus.detach()).sum()
        F_R = sc_plus + sc_minus
        sc_plus_star = self.s_plus.mul(c_plus_star.detach()).sum()
        sc_minus_star = self.s_minus.mul(c_minus_star.detach()).sum()
        F_R_star = sc_plus_star + sc_minus_star

        loss =  Delta_R_R_star + (F_R - F_R_star)/(self.P * self.N)

        # if loss < 0:
        if self.print:
            print('------------------------------------------')
            print('scores',scores)
            print('true_ranking', true_ranking)
            print('self.C',self.C)
            print('self.P',self.P)
            print('self.N',self.N)
            print('P*N',self.N*self.P)
            print('self.s_plus',self.s_plus)
            print('self.s_minus',self.s_minus)
            print('self.opt',self.opt)
            print('s_plus', self.s_plus, 's_minus', self.s_minus)
            print('c_plus, c_minus, c_plus_star, c_minus_star', c_plus, c_minus, c_plus_star, c_minus_star)
            print('sc_plus, sc_minus, sc_plus_star, sc_minus_star', sc_plus, sc_minus, sc_plus_star, sc_minus_star)
            print('F_R', F_R, 'F_R_star', F_R_star)
            print('loss', loss)
            print('Delta_R_R_star', Delta_R_R_star)
            print('------------------------------------------')
            input('press any key')

        return loss

    def forward(self, scores, class_lables):
        loss = 0
        for i in range(self.n_classes):
            loss += self.calc_loss_for_cth_class(scores[:,i], class_lables, i)
        return loss/self.n_classes

if __name__ == "__main__":
    rankloss = NDCG(1)
    loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([1,1,1,1,1,1,1,0,0,0]).long())
    print(loss)
    loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([1,1,1,1,1,1,0,1,0,0]).long())
    print(loss)
    loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([1,1,1,1,1,0,1,1,0,0]).long())
    print(loss)
    loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([1,1,1,1,1,0,0,1,1,0]).long())
    print(loss)
    loss = rankloss(torch.randn(10).view(-1,1), torch.tensor([1,1,1,1,1,0,0,1,1,0]).long())
    print(loss)
    # loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([0,0,0,9,9,9,9,9,9,9]).long())
    # print(loss)
    # loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1), torch.tensor([9,9,0,9,0,9,0,9,0,9]).long())
    # print(loss)
    # loss = rankloss(torch.tensor([0,2,4,6,8,10,12,14,16,18]).view(-1,1), torch.tensor([9,9,0,9,0,9,0,9,0,9]).long())
    # print(loss)


    # loss = rankloss(torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(-1,1)*1e-9, torch.tensor([9,9,0,9,0,9,0,9,0,9]).long())
    # print(loss)
    # loss = rankloss(torch.tensor([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]).view(-1,1), torch.tensor([9,9,0,9,0,9,0,9,0,9]).long())
    # print(loss)
    # loss = rankloss(torch.tensor([0,-1,-2,-3,-4,-5,-6,-7,-8,-9]).view(-1,1), torch.tensor([9,9,0,9,0,9,0,9,0,9]).long())

    # print(loss)
    # loss = rankloss(torch.tensor([1.7461, 0.2016, 0.5150, 1.5473]).view(-1,1), torch.tensor([0,9,0,9]).long())

    # loss = rankloss(torch.randn(4).view(-1,1), torch.tensor([0,0,9,9]).long())
    # loss = rankloss(torch.tensor([1,8,3,6,4,7,9,5,0,2]), torch.tensor([0,0,0,0,0,0,0,1,1,1]).long())
    # print(loss)


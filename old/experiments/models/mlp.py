import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, num_classes, width, depth, bias=True):

        super(MLP, self).__init__()
        n_in, n_out = 2, width
        if depth > 1:
            hidden = [nn.Linear(n_in, n_out, bias=bias), nn.ReLU()]
            for _ in range(depth - 2):
                n_in = n_out
                hidden.append(nn.Linear(n_in, n_out, bias=bias))
                hidden.append(nn.ReLU())
            hidden.append(nn.Linear(n_out, 2, bias=bias))
            hidden.append(nn.ReLU())
            self.features = nn.Sequential(*hidden)
        else:
            self.features = nn.Sequential(nn.Linear(2, 2, bias=True),
                                          nn.ReLU())
        self.clf = nn.Linear(2, 2, bias=bias)

    def forward(self, x):

        x = self.features(x)
        x = self.clf(x)
        return x



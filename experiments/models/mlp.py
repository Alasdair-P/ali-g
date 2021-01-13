import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, depth, num_classes, width, input_dims, bias=True):
        # print(depth, num_classes, width, input_dims, bias)

        super(MLP, self).__init__()
        self.input_dim = input_dims
        n_in, n_out = input_dims, width
        if depth == 0:
            self.features = nn.Linear(input_dims, num_classes, bias=bias)
        else:
            hidden = [nn.Linear(n_in, n_out, bias=bias), nn.ReLU()]
            for _ in range(depth-1):
                n_in = n_out
                hidden.append(nn.Linear(n_in, n_out, bias=bias))
                hidden.append(nn.ReLU())
            hidden.append(nn.Linear(n_out, num_classes, bias=bias))
            self.features = nn.Sequential(*hidden)

    def forward(self, x):
        x = x.view(-1,self.input_dim)
        x = self.features(x)
        return x

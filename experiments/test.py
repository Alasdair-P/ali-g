import torch
P = 10
s_minus = torch.tensor([1,3,5,8,8,9,11,11,11,11,11,11]).long()
i = torch.arange(P)
ri = torch.zeros(P)
for i in range(len(ri)):
    ri[i] = 1 + (s_minus <= (i+1)).long().sum()

ri_2 = 2 * torch.arange(1,P+1) + 1 - s_minus[0:P]
print(ri)
print(ri_2)

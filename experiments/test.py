import torch
import numpy as np
import time
print('hello world')
start_time = time.time()
a = torch.randn(100,100).cuda()
b = torch.randn(100,100).cuda()
c = a.mm(b).sum().cpu().numpy()
print('time taken', time.time()-start_time)
print('goodbye world')
np.save('/jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/test_please_delete', c)
print('cruel world')

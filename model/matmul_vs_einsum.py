import os
import torch
from time import time

# torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

# 1) fp32
a = torch.empty(24,40,32,48, dtype=torch.float32).to(device)
b = torch.empty(64,40,32,48, dtype=torch.float32).to(device)
c = torch.empty(40,48,24,32, dtype=torch.float32).to(device)
d = torch.empty(40,48,32,64, dtype=torch.float32).to(device)

torch.cuda.synchronize()
st = time()
for _ in range(1000):
    c.matmul(d)
torch.cuda.synchronize()
print(time()-st)

torch.cuda.synchronize()
st = time()
for _ in range(1000):
    torch.einsum('ibnd,jbnd->ijbn', a, b)
torch.cuda.synchronize()
print(time()-st)
#
# # 2) fp16
a = torch.empty(24,32,40,48, dtype=torch.float16).to(device)
b = torch.empty(64,32,40,48, dtype=torch.float16).to(device)
c = torch.empty(40,48,24,32, dtype=torch.float16).to(device)
d = torch.empty(40,48,32,64, dtype=torch.float16).to(device)

torch.cuda.synchronize()
st = time()
for _ in range(1000):
    torch.matmul(c,d)
torch.cuda.synchronize()
print(time()-st)

torch.cuda.synchronize()
st = time()
for _ in range(1000):
    torch.einsum('ibnd,jbnd->ijbn', a, b)
torch.cuda.synchronize()
print(time()-st)

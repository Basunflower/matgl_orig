import torch
import os

# torch.cuda.init()
os.environ['CUDA_VISIBLE_DEVICES']='1'
# torch.cuda.set_device(1)
print(torch.cuda.current_device())
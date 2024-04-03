import torch
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
torch.cuda.set_device(0)
print(torch.cuda.current_device())
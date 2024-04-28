import os
import torch

os.environ["dataset"] = "cifar10"
os.environ["epochs"] = "40"
os.environ["max_lr"] = 0.01
os.environ["grad_clip"] = 0.1
os.environ["weight_decay"] = 1e-4
os.environ["opt_func"] = torch.optim.Adam
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import faces_dataset

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

mae_loss = nn.L1Loss()
output = mae_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
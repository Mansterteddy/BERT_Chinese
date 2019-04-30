import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable

import os
import sys
import random
import time

from focal_loss import *

x = torch.rand(12800, 2) * random.randint(1, 10)
x = Variable(x)
print("x: ", x)

l = torch.rand(12800).ge(0.1).long()
l = Variable(l)
print("l: ", l)

output_0 = FocalLoss(gamma=0)(x, l)
output_1 = nn.CrossEntropyLoss()(x, l)
output_2 = FocalLoss(gamma=2, alpha=0.5)(x, l)

print("output_0: ", output_0)
print("output_1: ", output_1)
print("output_2: ", output_2)
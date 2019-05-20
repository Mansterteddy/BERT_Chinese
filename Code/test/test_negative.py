import torch
import torch.nn.functional as F

def forward(input):
    mask = torch.eye(3)
    loss = F.log_softmax(input, dim=1) * mask
    loss = (-loss.sum(dim=1)).mean()
    return loss

input = torch.FloatTensor([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
loss = forward(input)
print("loss: ", loss)
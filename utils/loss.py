import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def ldmk_loss(input, target, weight=None, size_average=True):
    n, c = input.size()

    loss_ = (input - target) ** 2
    iod = torch.sqrt(torch.sum(
        (target[:, 36*2:37*2] - target[:, 45*2:46*2])**2, 1))
    loss = torch.autograd.Variable(torch.zeros((n, c//2))).float().cuda()
    for i in range(c//2):
        loss[:, i] = torch.sqrt((loss_[:, i*2] + loss_[:, i*2+1])) / (iod+1e-6)

    if size_average:
        loss = torch.mean(loss)
    return loss

if __name__ == "__main__":
    pred = torch.zeros((4,136))
    gt = torch.ones((4,136))
    vpred = torch.autograd.Variable(pred, requires_grad=True).float().cuda()
    vgt = torch.autograd.Variable(gt).float().cuda()
    print(ldmk_loss(vpred, vgt))

import torch.nn as nn
import torch.nn.functional as F

class SElayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y.repeat(1, 1, w, h)
        return x * y


class attentionCRF(nn.Module):
    def __init__(self, inplanes, lrnsize, itersize, useparts=False):
        super(attentionCRF, self).__init__()
        self.inplanes = inplanes
        self.lrnsize = lrnsize
        self.pad = self.lrnsize // 2
        self.itersize = itersize
        self.useparts = useparts
        self.conv1 = nn.Conv2d(self.inplanes, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, self.lrnsize, 1, self.pad)
        if self.useparts:
            self.conv3 = nn.Conv2d(self.inplanes, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def _attention_foward(self, x):
        Q = []
        C = []
        conv = self.conv1(x)
        # RNN
        for i in range(self.itersize):
            if i == 0:
                conv2 = self.conv2(conv)
            else:
                conv2 = self.conv2(Q[i-1])

            C.append(conv2)
            tmp = self.sigmoid(C[i] + conv)
            Q.append(tmp)

        return x * Q[-1].repeat(self.inplanes, 2)


    def forward(self, x):
        if not self.useparts:
            return self._attention_foward(x)
        else:
            partnum = 68
            pre = []
            for i in range(68):
                att = self._attention_foward(x)
                s = self.conv3(x)
                pre.append(s)

        return torch.cat(p, 1)

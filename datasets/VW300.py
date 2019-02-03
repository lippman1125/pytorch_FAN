from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io
import copy

import torch
import torch.utils.data as data
# from torch.utils.serialization import load_lua
import torchfile

from utils.imutils import *
from utils.transforms import *

from datasets.W300LP import W300LP


class VW300(W300LP):

    def __init__(self, args, split):
        super(VW300, self).__init__(args, split)
        assert self.pointType == '3D', "Only 3D face alignment supported for now"

    def _getCategory(self, split):
        return {
            'train': "TrainSet",
            'A': "CatA",
            'B': "CatB",
            'C': "CatC",
        }[split]

    def _getDataFaces(self, is_train):
        # split = self._getCategory(self.split)
        lines = []
        if is_train:
            pass
        else:
            for split in ['CatA', 'CatB', 'CatC']:
                base_dir = os.path.join(self.img_folder, split)
                dirs = os.listdir(base_dir)
                for d in dirs:
                    files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.t7')]
                    for f in files:
                        lines.append(os.path.join(base_dir, d, f))
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
        else:
            print('=> loaded 300VW-3D dataset, {} images were found'.format(len(lines)))
        return lines

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = torchfile.load(self.anno[idx])
        pts = main_pts  # 3D landmarks only. # if self.pointType == '2D' else main_pts[1]
        mins_ = torch.min(pts, 0)[0].view(2)  # min vals
        maxs_ = torch.max(pts, 0)[0].view(2)  # max vals
        c = torch.FloatTensor((maxs_[0] - (maxs_[0] - mins_[0]) / 2,
                               maxs_[1] - (maxs_[1] - mins_[1]) / 2))
        c[1] -= ((maxs_[1] - mins_[1]) * 0.12)
        s = (maxs_[0] - mins_[0] + maxs_[1] - mins_[1]) / 195

        img = load_image(self.anno[idx][:-3] + '.jpg')

        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='vw300')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = crop(img, c, s, [256, 256], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        tpts = cooy.deepcopy(pts)
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [64, 64], rot=r))
                out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, out, pts, c, s

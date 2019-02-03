from __future__ import print_function

import matplotlib
import os
import numpy as np
import random
import math
from skimage import io
import copy
import cv2

import torch

# from torch.utils.serialization import load_lua
import torchfile

# from utils.utils import *
from utils.imutils import *
from utils.transforms import *

from datasets.W300LP import W300LP


class LS3DW(W300LP):

    def __init__(self, args, split):
        super(LS3DW, self).__init__(args, split)
        assert self.pointType == '3D'

    def _getDataFaces(self, is_train):
        base_dir = os.path.join(self.img_folder, 'new_dataset')
        E, M, H = [], [], []
        vallines = []
        lines = []
        files = [f for f in os.listdir(base_dir) if f.endswith('.t7')]
        for f in files:
            num_of_file = int(f.split('.')[0])
            if num_of_file % 3 == 1:  # 0-30
                E.append(os.path.join(base_dir, f))
            elif num_of_file % 3 == 2:  # 30-60
                M.append(os.path.join(base_dir, f))
            else:  # 60-90
                H.append(os.path.join(base_dir, f))
        vallines.extend(E)
        vallines.extend(M)
        vallines.extend(H)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = torchfile.load(self.anno[idx])
        pts = main_pts
        mins_ = torch.min(torch.from_numpy(pts).float(), 0)[0].view(2)  # min vals
        maxs_ = torch.max(torch.from_numpy(pts).float(), 0)[0].view(2)  # max vals
        # print(mins_)
        # print(maxs_)
        c = torch.FloatTensor((maxs_[0] - (maxs_[0] - mins_[0]) / 2,
                               maxs_[1] - (maxs_[1] - mins_[1]) / 2))
        # c[0] -= ((maxs_[0] - mins_[0]) * 0.12)
        c[1] -= ((maxs_[1] - mins_[1]) * 0.12)
        s = (maxs_[0] - mins_[0] + maxs_[1] - mins_[1]) / 195

        img = load_image(self.anno[idx][:-3] + '.jpg')

        r = 0
        if self.is_train:
            # scale
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            # rotatation
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0
            # flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                c[0] = img.size(2) - c[0]
            # RGB
            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = crop(img, c, s, [256, 256], rot=r)
        # inp = color_normalize(inp, self.mean, self.std)

        if self.is_train:
            tpts = copy.deepcopy(pts)
            out = torch.zeros(self.nParts, 64, 64)
            for i in range(self.nParts):
                if tpts[i, 0] > 0:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [64, 64], rot=r))
                    out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)
        else:
            tpts = copy.deepcopy(pts)
            out = torch.zeros(self.nParts, 256, 256)
            for i in range(self.nParts):
                if tpts[i, 0] > 0:
                    tpts[i, 0:2] = transform(tpts[i, 0:2] + 1, c, s, [256, 256], rot=r)
                    out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, out, tpts, c, s

    def _comput_mean(self):
        meanstd_file = './data/300W_LP/mean.pth.tar'
        if os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print(
                "\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe..."
            )
            mean = torch.zeros(3)
            std = torch.zeros(3)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_folder, self.anno[i].split('_')[0],
                                            self.anno[i][:-8] + '.jpg')
                    img = load_image(img_path)
                    mean += img.view(img.size(0), -1).mean(1)
                    std += img.view(img.size(0), -1).std(1)

            mean /= self.total
            std /= self.total
            ms = {
                'mean': mean,
                'std': std,
            }
            torch.save(ms, meanstd_file)
        if self.is_train:
            print('\tMean: %.4f, %.4f, %.4f' % (ms['mean'][0], ms['mean'][1], ms['mean'][2]))
            print('\tStd:  %.4f, %.4f, %.4f' % (ms['std'][0], ms['std'][1], ms['std'][2]))
        return ms['mean'], ms['std']

if __name__=="__main__":
    import opts
    args = opts.argparser()
    args.data = "data/LS3D-W"
    args.pointType = '3D'
    dataset = LS3DW(args, 'test')
    crop_win = None
    for i in range(dataset.__len__()):
        input, target, meta = dataset.__getitem__(i)
        # input = input.numpy().transpose(1,2,0)
        # target = target.numpy()
        # if crop_win is None:
        #     crop_win = plt.imshow(input)
        # else:
        #     crop_win.set_data(input)
        # plt.pause(1)
        # plt.draw
    # gts, gtfiles = demo.loadgts(args.data, args.pointType)
    # for i in range(len(gtfiles)):
    #     if not gtfiles[i] == dataset.anno[i]:
    #         print(gtfiles[i], dataset.anno[i])
    #         exit()
    # print("All file are same")
        input = input.numpy().transpose(1, 2, 0) * 255.
        target = target.numpy().transpose(1, 2, 0) * 255
        input = np.ascontiguousarray(input, dtype=np.uint8)
        target = np.ascontiguousarray(target, dtype=np.uint32)

        # print(np.shape(target))
        pts = meta["pts"].astype(np.uint8)
        # print(pts)

        # print(np.shape(input))
        # print(input.dtype)
        for i in range(68):
            cv2.circle(input, (pts[i][0], pts[i][1]), 3, (0, 0, 255), 2)

        cv2.imshow("face", input[:, :, ::-1])
        # for i in range(68):
        #     cv2.imshow("heatmap", target[:,:,i])
        #     cv2.waitKey(0)
        target = np.sum(target, axis=2, keepdims=True)
        cv2.imshow("heatmap", target.astype(np.uint8))
        cv2.waitKey(30)

from __future__ import print_function

import os
import sys
import numpy as np
import random
import math
from skimage import io

import torch
import torch.utils.data as data
# from torch.utils.serialization import load_lua
import torchfile
import copy
import cv2

# sys.path.append("../")
# from utils.utils import *
from utils.imutils import *
from utils.transforms import *


class W300LP(data.Dataset):

    def __init__(self, args, split):
        self.nParts = 68
        self.pointType = args.pointType
        # self.anno = anno
        self.img_folder = args.data
        self.split = split
        self.is_train = True if self.split == 'train' else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor
        self.mean, self.std = self._comput_mean()

    def _getDataFaces(self, is_train):
        base_dir = os.path.join(self.img_folder, 'landmarks')
        dirs = os.listdir(base_dir)
        lines = []
        vallines = []
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.t7')]
            for f in files:
                if f.find('test') == -1:
                    lines.append(f)
                else:
                    vallines.append(f)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, c, s = self.generateSampleFace(index)
        self.pts, self.c, self.s = pts, c, s
        if self.is_train:
            return inp, out
        else:
            meta = {'index': index, 'center': c, 'scale': s, 'pts': pts,}
            return inp, out, meta

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        # main_pts = load_lua(
        #     os.path.join(self.img_folder, 'landmarks', self.anno[idx].split('_')[0],
        #                  self.anno[idx][:-4] + '.t7'))
        main_pts = torchfile.load(
            os.path.join(self.img_folder, 'landmarks', self.anno[idx].split('_')[0],
                         self.anno[idx]))
        pts = main_pts[0] if self.pointType == '2D' else main_pts[1]
        c = torch.Tensor((450 / 2, 450 / 2 + 50))
        s = 1.8

        # print(os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-7] +'.jpg'))
        img = load_image(
            os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-7] +
                         '.jpg'))

        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = crop(img, c, s, [256, 256], rot=r)
        # inp = color_normalize(inp, self.mean, self.std)

        #if self.is_train:
        tpts = copy.deepcopy(pts)
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [64, 64], rot=r))
                out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)
        #else:
        #    tpts = copy.deepcopy(pts)
        #    out = torch.zeros(self.nParts, 256, 256)
        #    for i in range(self.nParts):
        #        if tpts[i, 0] > 0:
        #            tpts[i, 0:2] = transform(tpts[i, 0:2] + 1, c, s, [256, 256], rot=r)
        #            out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, out, tpts, c, s

    def _comput_mean(self):
        meanstd_file = './data/300W_LP/mean.pth.tar'
        if os.path.exists(meanstd_file) and os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print("\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe...")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_folder, self.anno[i].split('_')[0],
                                            self.anno[i][:-7] + '.jpg')
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

    mode = "test"

    if mode == "test":
        dataset = W300LP(args, mode)
        crop_win = None
        for i in range(dataset.__len__()):
            input, target, meta = dataset.__getitem__(i)
            input = input.numpy().transpose(1,2,0) * 255.
            target = target.numpy().transpose(1, 2, 0) * 255
            input = np.ascontiguousarray(input, dtype=np.uint8)
            target = np.ascontiguousarray(target, dtype=np.uint32)

            print(np.shape(target))
            pts = meta["pts"].astype(np.uint8)
            # print(pts)

            # print(np.shape(input))
            print(input.dtype)
            for i in range(68):
                cv2.circle(input, (pts[i][0], pts[i][1]), 3, (0, 0, 255), 2)

            cv2.imshow("face", input[:,:,::-1])
            cv2.waitKey(0)
            # for i in range(68):
            #     cv2.imshow("heatmap", target[:,:,i])
            #     cv2.waitKey(0)
            target = np.sum(target, axis=2, keepdims=True)
            cv2.imshow("heatmap", target.astype(np.uint8))
            cv2.waitKey(0)

    elif mode == "train":
        dataset = W300LP(args, mode)
        crop_win = None
        for i in range(dataset.__len__()):
            input, target = dataset.__getitem__(i)
            input = input.numpy().transpose(1, 2, 0) * 255.0
            target = target.numpy().transpose(1, 2, 0) * 255.0
            input = np.ascontiguousarray(input, dtype=np.uint8)
            target = np.ascontiguousarray(target, dtype=np.uint32)
            # print(np.shape(input))
            cv2.imshow("face", input[:, :, ::-1])
            cv2.waitKey(0)
            # for i in range(68):
            #     cv2.imshow("heatmap", target[:, :, i])
            #     cv2.waitKey(0)
            # for k in range(68):
            #     for m in range(64):
            #         for n in range(64):
            #             print(target[m,n,k], end=" ")
            #         print("\n")
            #     print("-------------------------------------")
            target = np.sum(target, axis=2, keepdims=True)
            # print(np.shape(target))
            # for m in range(64):
            #     for n in range(64):
            #         print(target[m, n, 0], end=" ")
            #     print("\n")
            cv2.imshow("heatmap", target.astype(np.uint8))
            cv2.waitKey(0)

        # if crop_win is None:
        #     crop_win = plt.imshow(input)
        # else:
        #     crop_win.set_data(input)
        # plt.pause(0.5)
        # plt.draw

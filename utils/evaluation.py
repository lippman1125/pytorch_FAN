from __future__ import absolute_import, print_function

import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy', 'AverageMeter']


def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    # batch, chn, height, width ===> batch, chn, height*width
    # chn = 68
    # height*width = score_map
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    # batchsize * numPoints * 2
    # 0 is x coord
    # 1 is y coord
    # shape = batchsize, numPoints, 2
    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    # dists = 68 x batch
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum() * 1.0 / dists.ne(-1).sum()
    else:
        return -1


def calc_metrics(dists, path='', category=''):
    errors = torch.mean(dists, 0).view(dists.size(1))
    axes1 = np.linspace(0, 1, 1000)
    axes2 = np.zeros(1000)
    for i in range(1000):
        axes2[i] = float((errors < axes1[i]).sum()) / float(errors.size(0))

    auc = round(np.sum(axes2[:70]) / .7, 2)

    if path:
        label = '{}({}) : {}'.format(path.split('/')[2], category, str(auc))
        plt.xlim(0, 7)
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 110, 10))
        plt.xticks(np.arange(0, 8, 1))

        plt.grid()
        plt.title('NME (%)', fontsize=20)
        plt.xlabel('NME (%)', fontsize=16)
        plt.ylabel('Test images (%)', fontsize=16)
        if category:
            if category in ['Easy', 'Category A']:
                plt.plot(axes1 * 100, axes2 * 100, 'b-', label=label, lw=3)
            if category in ['Media', 'Category B']:
                plt.plot(axes1 * 100, axes2 * 100, 'r-', label=label, lw=3)
            if category in ['Hard', 'Category C']:
                plt.plot(axes1 * 100, axes2 * 100, 'g-', label=label, lw=3)
        else:
            plt.plot(axes1 * 100, axes2 * 100, 'b-', label=label, lw=3)
        plt.legend(loc=4, fontsize=12)

        plt.savefig(os.path.join(path + '/CED.eps'))
    return auc


def _get_bboxsize(iterable):
    # iterable = 68 x 2
    # torch.min return values, idxs
    mins = torch.min(iterable, 0)[0].view(2)
    maxs = torch.max(iterable, 0)[0].view(2)

    center = torch.FloatTensor((maxs[0] - (maxs[0] - mins[0]) / 2,
                                maxs[1] - (maxs[1] - mins[1]) / 2))
    # center[1] = center[1] - ((maxs[1] - mins[1]) * 0.12)

    return np.sqrt(abs(maxs[0] - mins[0]) * abs(maxs[1] - mins[1]))


def accuracy(output, target, idxs, thr=0.08):
    ''' Calculate accuracy according to NME, but uses ground truth heatmap rather than x,y locations
    First value to be returned is accuracy calculated based on overall 'idxs'
    followed by individual accuracies
    '''
    # preds = batch, 68, 64, 64
    preds = get_preds(output)
    gts = get_preds(target)
    # B * 2
    norm = torch.ones(preds.size(0))
    # use face bbox to normalize
    for i, gt in enumerate(gts):
        norm[i] = _get_bboxsize(gt)

    dists = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0

    mean_dists = torch.mean(dists, 0)
    acc[0] = mean_dists.le(thr).sum() * 1.0 / preds.size(0)
    # for i in range(len(idxs)):
    #     acc[i+1] = dist_acc(dists[idxs[i]-1], thr=thr)
    #     if acc[i+1] >= 0:
    #         avg_acc = avg_acc + acc[i+1]
    #         cnt += 1

    # if cnt != 0:
    #     acc[0] = avg_acc / cnt
    return acc, dists


def final_preds(output, center, scale, res):
    if output.size(1) == 136:
        coords = output.view((output.szie(0), 68, 2))
    else:
        coords = get_preds(output)  # float type

    # output shape is batch, 68, 64, 64
    # coords shape is batch, 68, 2
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor(
                    [hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())
    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

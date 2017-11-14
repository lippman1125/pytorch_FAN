import os
import torch
from torch.utils.serialization import load_lua
import numpy as np
import os.path as osp
import scipy.io as sio

from utils.evaluation import calc_dists, calc_metrics, _get_bboxsize


def loadpreds_if_exists(path):
    if not os.path.isfile(path):
        print("WARNING: predictions do not exist!!! considering to run 'python main.py -e'")
        exit
    else:
        return sio.loadmat(path)['preds']

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def loadgts():
    base_dir = os.path.join('./data/300W_LP', 'landmarks')
    dirs = os.listdir(base_dir)
    print(dirs)
    lines = []
    for d in dirs:
        files = [
            f for f in os.listdir(osp.join(base_dir, d))
            if f.endswith('mat') and f.find('test') != -1
        ]
        lines.extend(files)

    print(len(lines))
    all_gts = np.zeros((len(lines), 68, 2))
    for i, f in enumerate(lines):
        pts = load_lua(osp.join(base_dir, f.split('_')[0], f[:-4]+'.t7'))[0].numpy()
        all_gts[i, :, :] = pts
    return all_gts

if __name__=="__main__":
    import opts
    args = opts.argparser()
    preds = torch.from_numpy(loadpreds_if_exists(osp.join(args.checkpoint, 'preds_valid.mat')))
    gts = torch.from_numpy(loadgts())
    norm = np.ones(preds.size(0))
    for i, gt in enumerate(gts):
        norm[i] = _get_bboxsize(gt)

    dists = calc_dists(preds, gts, norm)
    auc = calc_metrics(dists, True)
    print("Mean Error: {}. AUC: {}".format(torch.mean(dists), auc))

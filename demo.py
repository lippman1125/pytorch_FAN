import os
import torch
import matplotlib
matplotlib.use('Agg')
from torch.utils.serialization import load_lua
import numpy as np
import os.path as osp
import scipy.io as sio

from utils.evaluation import calc_dists, calc_metrics, _get_bboxsize


def loadpreds_if_exists(path):
    if not os.path.isfile(path):
        print(path)
        print("FATAL ERROR: predictions do not exist!!! considering to run 'python main.py -e'")
        exit()
    else:
        preds = sio.loadmat(path)['preds']
        return sio.loadmat(path)['preds']


def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def loadgts(datapath, pointType='2D'):
    if datapath.endswith('300W_LP'):
        base_dir = os.path.join(datapath, 'landmarks')
        dirs = os.listdir(base_dir)
        lines = []
        for d in dirs:
            files = [
                f for f in os.listdir(osp.join(base_dir, d))
                if f.endswith('mat') and f.find('test') != -1
            ]
            lines.extend(files)
        all_gts = torch.zeros((len(lines), 68, 2))
        for i, f in enumerate(lines):
            if pointType == '2D':
                pts = load_lua(osp.join(base_dir, f.split('_')[0], f[:-4] + '.t7'))[0]
            else:
                pts = load_lua(osp.join(base_dir, f.split('_')[0], f[:-4] + '.t7'))[1]
            all_gts[i, :, :] = pts

    else:
        base_dir = os.path.join(datapath, 'CatA')
        dirs = os.listdir(base_dir)
        lines = []
        for d in dirs:
            files = [
                osp.join(base_dir, d, f) for f in os.listdir(osp.join(base_dir, d))
                if f.endswith('t7')
            ]
            lines.extend(files)
        all_gts = torch.zeros((len(lines), 68, 2))
        for i, f in enumerate(lines):
            if pointType == '2D':
                print("300VW-3D do not have 3D annotations.")
                exit()
            else:
                pts = load_lua(f)
            all_gts[i, :, :] = pts
        print('Loaded {} sample from {}'.format(len(lines), base_dir))

    return all_gts


if __name__ == "__main__":
    import opts
    args = opts.argparser()
    dataset = args.data.split('/')[-1]
    save_dir = osp.join(args.checkpoint, dataset)
    preds = torch.from_numpy(loadpreds_if_exists(osp.join(save_dir, 'preds_valid.mat')))
    gts = loadgts(args.data, args.pointType)
    norm = np.ones(preds.size(0))
    for i, gt in enumerate(gts):
        norm[i] = _get_bboxsize(gt)

    dists = calc_dists(preds, gts, norm)
    auc = calc_metrics(dists, save_dir)
    print("Mean Error: {}. AUC: {}".format(round(torch.mean(dists) * 100., 2), auc))

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
        return all_gts, lines

    elif datapath.find('300VW-3D') != -1:
        lines = []
        for split in ['CatA', 'CatB', 'CatC']:
            base_dir = os.path.join(datapath, split)
            dirs = os.listdir(base_dir)
            for d in dirs:
                files = [
                    osp.join(base_dir, d, f) for f in os.listdir(osp.join(base_dir, d))
                    if f.endswith('t7')
                ]
                lines.extend(files)
    elif datapath.endswith('LS3D-W'):
        base_dir = osp.join(datapath, 'new_dataset')
        lines, E, M, H = [],[],[],[]
        files = [f for f in os.listdir(base_dir) if f.endswith('.t7')]
        for f in files:
            num_of_file = int(f.split('.')[0])
            if num_of_file % 3 == 1:  # 0-30
                E.append(os.path.join(base_dir, f))
            elif num_of_file % 3 == 2:  # 30-60
                M.append(os.path.join(base_dir, f))
            else:  # 60-90
                H.append(os.path.join(base_dir, f))
        lines.extend(E)
        lines.extend(M)
        lines.extend(H)

    all_gts = torch.zeros((len(lines), 68, 2))
    for i, f in enumerate(lines):
        if pointType == '2D':
            if datapath.endswith('300W_LP'):
                pts = load_lua(osp.join(base_dir, f.split('_')[0], f[:-4] + '.t7'))[0]
            else:
                print("Given data set do not have 3D annotations.")
                exit()
        else:
            pts = load_lua(f)
        all_gts[i, :, :] = pts
    print('Loaded {} sample from {}'.format(len(lines), base_dir))

    return all_gts, lines


if __name__ == "__main__":
    import opts
    args = opts.argparser()
    dataset = args.data.split('/')[-1]
    save_dir = osp.join(args.checkpoint, dataset)
    print("save dictory: " + save_dir)
    preds = torch.from_numpy(loadpreds_if_exists(osp.join(save_dir, 'preds_valid.mat')))
    gts, _ = loadgts(args.data, args.pointType)
    norm = np.ones(preds.size(0))
    for i, gt in enumerate(gts):
        norm[i] = _get_bboxsize(gt)

    if dataset == 'LS3D-W' or dataset == '300VW-3D':
        for i in range(3):
            if dataset == 'LS3D-W':
                category = {'0': 'Easy', '1': 'Media', '2': 'Hard'}[str(i)]
                l, f = 2400*i, 2400*(i+1)
            else:
                category = {'0': 'Category A', '1': 'Category B', '2': 'Category C'}[str(i)]
                l, f = {0: [0, 62643], 1: [62643, 62642+32872], 2: [95515,-1]}[i]
            # For LS3D-W dataset which landmark indexed on `0`
            dist = calc_dists(preds[l:f] - 1., gts[l:f], norm[l:f])
            auc = calc_metrics(dist, save_dir, category)
            print("FINAL: Mean Error: {}. AUC: {} of {} subset".format(round(torch.mean(dist) * 100., 2), auc, category))
    else:
        dists = calc_dists(preds, gts, norm)
        auc = calc_metrics(dists, save_dir)
        print("FINAL: Mean Error: {}. AUC: {}".format(round(torch.mean(dists) * 100., 2), auc))

import os
import cv2
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import copy

from datasets import W300LP, VW300, AFLW2000, LS3DW
import models
from models.fan_model import FAN

from utils.evaluation import get_preds

CHECKPOINT_PATH = "./checkpoint_4Module/fan3d_wo_norm_att/model_best.pth.tar"

# # print(models.__dict__)
# a = [name for name in models.__dict__]
# print(a)


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

print(model_names)

netType = "fan"
nStacks = 2
nModules = 1
nFeats = 128
use_se = False
use_attention = True

class Arg():
    def __init__(self):
        self.data = 'data/LS3DW'
        self.pointType = '3D'

# model = models.__dict__[netType](
#     num_stacks=nStacks,
#     num_blocks=nModules,
#     num_feats=nFeats,
#     use_se = use_se,
#     use_attention=use_attention,
#     num_classes=68)
model = FAN(4)


model_dict = model.state_dict()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage)['state_dict']
for k in checkpoint.keys():
    model_dict[k.replace('module.', '')] = checkpoint[k]
model.load_state_dict(model_dict)

# print(checkpoint)
# exit()

# model = torch.nn.DataParallel(model)
# model.load_state_dict(checkpoint['state_dict'])
model.eval()
# epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']
#
# print(epoch)
# print(best_acc)
# print(model)

args = Arg()

args.data = "data/LS3D-W"
args.pointType = '3D'
args.scale_factor = 0.3
args.rot_factor = 30

# if pin_memory = True, dataloader will copy data into cuda mem
# test_loader = torch.utils.data.DataLoader(
#     LS3DW(args, 'test'),
#     batch_size=1,
#     shuffle=False)

# with torch.no_grad():
#     for i, (inputs, target, meta) in enumerate(test_loader):
#
#         # print(type(inputs))
#         # input_var = torch.autograd.Variable(inputs)
#         # target_var = torch.autograd.Variable(target)
#         print(type(inputs))
#         print(inputs.size())
#         inputs = inputs.to(device)
#         target = target.to(device)
#         print(inputs.dtype)
#
#         output = model(inputs)
#         score_map = output[-1].data
#         print(score_map)
#         exit()
#
#         cv2.waitKey(0)


# test_dataset = LS3DW(args, 'test')
# crop_win = None
# for i in range(test_dataset.__len__()):
#     input, target, meta = test_dataset.__getitem__(i)
#     # print(type(inputs))
#     # input_var = torch.autograd.Variable(inputs)
#     # target_var = torch.autograd.Variable(target)
#     print(input.type())
#     input_ = input.unsqueeze(0)
#     print(type(input_))
#     print(input_.size())
#     # input = input.to(device)
#     # target = target.to(device)
#     print(input_.type())
#
#     output = model(input_)
#     score_map = output[-1].data
#
#     print(score_map.size())
#
#     upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)
#
#     score_map_4 = upsample(score_map)
#     score_map_4 = get_preds(score_map_4)
#
#     print(score_map_4.size())
#     score_map_4 = np.squeeze(score_map_4.numpy())
#     print(score_map_4)
#
#     # score_map = get_preds(score_map)
#     #
#     # print(score_map.size())
#     # score_map = np.squeeze(score_map.numpy())
#     # print(score_map)
#     # print(score_map.data)
#     # print(score_map.grad)
#     # print(score_map.grad_fn)
#
#     input = np.transpose(input.numpy(), (1,2,0)) * 255
#     input = np.ascontiguousarray(input, dtype=np.uint8)
#
#     # input = cv2.resize(input, (64, 64), interpolation=cv2.INTER_LINEAR)
#     ori_pts = meta["pts"]
#     for i in range(68):
#         cv2.circle(input, (int(score_map_4[i][0]), int(score_map_4[i][1])), 2, (255, 0, 0), -1)
#         # cv2.circle(input, (int(ori_pts[i][0]), int(ori_pts[i][1])), 2, (0, 255, 0), 1)
#     # input = cv2.resize(input, (256, 256))
#     cv2.imshow("orig", input[:, :, ::-1])
#     cv2.waitKey(0)



# img = cv2.imread("../face-alignment-pytorch/crop_0.jpg")
# img = cv2.imread("../video_neg_nocluster/1545615288538.mp4_5_4.798749_noglass_26_0_0.237937.jpg")
img = cv2.imread("crop_1.jpg")
img = cv2.resize(img, (256, 256))
img_trans = np.transpose(img, (2,0,1)).astype(np.float32)
img_trans2 = copy.deepcopy(img_trans[::-1,:,:])
img_float = img_trans2 / 255.0

input = torch.from_numpy(img_float)
# print(type(inputs))
# input_var = torch.autograd.Variable(inputs)
# target_var = torch.autograd.Variable(target)
print(input.type())
input_ = input.unsqueeze(0)
print(type(input_))
print(input_.size())
# input = input.to(device)
# target = target.to(device)
print(input_.type())

output = model(input_)
score_map = output[-1].data

print(score_map.size())

upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)

score_map_4 = upsample(score_map)
score_map_4 = get_preds(score_map_4)

print(score_map_4.size())
score_map_4 = np.squeeze(score_map_4.numpy())
print(score_map_4)
# print(score_map.data)
# print(score_map.grad)
# print(score_map.grad_fn)

input = np.transpose(input.numpy(), (1, 2, 0)) * 255
input = np.ascontiguousarray(input, dtype=np.uint8)

# input = cv2.resize(input, (64, 64), interpolation=cv2.INTER_LINEAR)
# ori_pts = meta["pts"]
for i in range(68):
    cv2.circle(img, (int(score_map_4[i][0]), int(score_map_4[i][1])), 2, (0, 0, 255), 1)
    # cv2.circle(input, (int(ori_pts[i][0]), int(ori_pts[i][1])), 2, (0, 255, 0), 1)
# input = cv2.resize(input, (256, 256))
cv2.imshow("orig", img)
cv2.waitKey(0)
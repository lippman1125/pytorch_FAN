import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import copy

from utils.imutils import *
from utils.transforms import *
from datasets import W300LP, VW300, AFLW2000, LS3DW
import models
from models.fan_model import FAN
from utils.evaluation import get_preds, final_preds
from faceboxes import face_detector_init, detect

CHECKPOINT_PATH = "./checkpoint/fan3d_wo_norm_att/model_best.pth.tar"


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

model = FAN(2)
# model = torch.nn.DataParallel(model).cuda()


model_dict = model.state_dict()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage)['state_dict']
for k in checkpoint.keys():
    model_dict[k.replace('module.', '')] = checkpoint[k]
model.load_state_dict(model_dict)

model.eval()

proto = "faceboxes_deploy.prototxt"
mdl = "faceboxes_iter_120000.caffemodel"
face_detector = face_detector_init(proto, mdl)

# def parse_roi_box_from_bbox(bbox):
#     left, top, right, bottom = bbox
#     old_size = (right - left + bottom - top) / 2
#     center_x = right - (right - left) / 2.0
#     center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
#     size = int(old_size * 1.58)
#     roi_box = [0] * 4
#     roi_box[0] = center_x - size / 2
#     roi_box[1] = center_y - size / 2
#     roi_box[2] = roi_box[0] + size
#     roi_box[3] = roi_box[1] + size
#     return roi_box
#
# def crop_img(img, roi_box):
#     h, w = img.shape[:2]
#
#     sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
#     dh, dw = ey - sy, ex - sx
#     if len(img.shape) == 3:
#         res = np.zeros((dh, dw, 3), dtype=np.uint8)
#     else:
#         res = np.zeros((dh, dw), dtype=np.uint8)
#     if sx < 0:
#         sx, dsx = 0, -sx
#     else:
#         dsx = 0
#
#     if ex > w:
#         ex, dex = w, dw - (ex - w)
#     else:
#         dex = dw
#
#     if sy < 0:
#         sy, dsy = 0, -sy
#     else:
#         dsy = 0
#
#     if ey > h:
#         ey, dey = h, dh - (ey - h)
#     else:
#         dey = dh
#
#     res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
#     return res

reference_scale = 200
cap = cv2.VideoCapture(0)
while True:
    _, img_ori = cap.read()

    # rects = face_detector(img_ori, 1)
    rects = detect(img_ori, face_detector)

    if len(rects) == 0:
        continue

    print(rects)

    for rect in rects:
        d = [rect.left() - 10, rect.top() - 10, rect.right() + 10, rect.bottom() + 10]
        # d = [rect.left() , rect.top() , rect.right() , rect.bottom()]

        center = torch.FloatTensor([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
        center[1] = center[1] + (d[3] - d[1]) * 0.12
        hw = max(d[2] - d[0], d[3] - d[1])
        scale = float(hw / reference_scale)
        print(scale)

        img_chn = copy.deepcopy(img_ori[:,:,::-1])
        img_trans = np.transpose(img_chn, (2,0,1))
        inp = crop(img_trans, center, scale)
        inp.unsqueeze_(0)

        output = model(inp)
        score_map = output[-1].data

        pts_img = final_preds(score_map, [center], [scale], [64, 64])

        # print(pts_img)
        pts_img = np.squeeze(pts_img.numpy())
        # print(pts_img)

        for i in range(pts_img.shape[0]):
            pts = pts_img[i]
            cv2.circle(img_ori, (pts[0], pts[1]), 2, (0, 255, 0), -1, 2)
        cv2.rectangle(img_ori, (d[0], d[1]), (d[2], d[3]), (255, 255, 255))


    cv2.imshow("landmark", img_ori)
    cv2.waitKey(5)
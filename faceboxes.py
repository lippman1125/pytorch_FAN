# -*- coding: utf-8 -*
import numpy as np  
import sys,os  
import cv2
import argparse
import caffe  
import time
import dlib

# class rect_t:
#     def __init__(self, left, top, right, bottom):
#         self.left = left
#         self.top  = top
#         self.right = right
#         self.bottom = bottom


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

# def detect(img_dir, imgfile):
#
#     full_path = os.path.join(img_dir, imgfile + ".jpg")
#     frame = cv2.imread(full_path)
#     transformed_image = transformer.preprocess('data', frame)
#
#     # transformed_image = frame.astype(np.float32) - [127.5, 127.5, 127.5]
#     # transformed_image = transformed_image / 128.0
#     # transformed_image = np.transpose(transformed_image, (2,0,1))
#     # transformed_image = transformed_image[np.newaxis, :,:,:]
#
#     # print img
#     # print(transformed_image)
#     # print(net.blobs['data'].data.shape)
#     # print(transformed_image.shape)
#
#
#     # net.blobs['data'].reshape(*(transformed_image.shape))
#     net.blobs['data'].data[...] = transformed_image
#
#     # print(net.blobs['data'].data.shape)
#     # exit()
#
#
#     time_start=time.time()
#     out = net.forward()
#     time_end=time.time()
#     print (time_end-time_start)
#     #print(out['detection_out'])
#     box, conf, cls = postprocess(frame, out)
#
#     count = 0
#     _str = ""
#     str_name = imgfile + "\n"
#     str_box = ""
#
#     _str += str_name
#     for i in range(len(box)):
#         p1 = (box[i][0], box[i][1])
#         p2 = (box[i][2], box[i][3])
#         if conf[i] >= 0.9 :
#             cv2.rectangle(frame, p1, p2, (0,255,0))
#             p3 = (max(p1[0], 15), max(p1[1], 15))
#             title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
#             cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
#
#         str_box += str(box[i][0]) + " " \
#              + str(box[i][1]) + " " \
#              + str(box[i][2] - box[i][0]) + " " \
#              + str(box[i][3] - box[i][1]) + " " \
#              + str(conf[i]) + "\n"
#         count += 1
#     _str += str(count) + "\n"
#     _str += str_box
#     print(_str)
#     return _str, frame


def detect(img, net):

    transformed_image = img.astype(np.float32) - [104, 117, 123]
    transformed_image = np.transpose(transformed_image, (2,0,1))
    transformed_image = transformed_image[np.newaxis, :,:,:]

    # print img
    # print(transformed_image)
    # print(net.blobs['data'].data.shape)
    # print(transformed_image.shape)


    net.blobs['data'].reshape(*(transformed_image.shape))
    net.blobs['data'].data[...] = transformed_image

    # print(net.blobs['data'].data.shape)
    # exit()


    time_start=time.time()
    out = net.forward()
    time_end=time.time()
    print (time_end-time_start)
    #print(out['detection_out'])
    box, conf, cls = postprocess(img, out)


    rects = []
    for i in range(len(box)):
        if conf[i] >= 0.9 :
            rect = dlib.rectangle(box[i][0], box[i][1], box[i][2], box[i][3])
            rects.append(rect)

    return rects

def face_detector_init(proto, model):

    caffe.set_mode_cpu()
    net = caffe.Net(proto, model, caffe.TEST)

    return net




# if __name__ == '__main__':
#     args = parse_arguments(sys.argv[1:])
#
#     image_dir = args.image_dir
#     file_list = args.file_list
#     file_result = args.file_result
#     proto = args.prototxt
#     model = args.caffemodel
#
#     if not os.path.exists(image_dir):
#         print("image_dir: {} does not exist".format(image_dir))
#         exit()
#     if not os.path.exists(file_list):
#         print("file_list: {} does not exist".format(file_list))
#         exit()
#     if not os.path.exists(proto):
#         print("prototxt: {} does not exist".format(proto))
#         exit()
#     if not os.path.exists(model):
#         print("caffemodel: {} does not exist".format(model))
#         exit()
#
#     output_dir = os.path.basename(image_dir) + "square_real_out_035_topk200_cuda"
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#
#     caffe.set_mode_gpu()
#     net = caffe.Net(proto, model,caffe.TEST)
#
#     CLASSES = ('background','face')
#
#     transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#     transformer.set_transpose('data', (2, 0, 1))
#     transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
#
#     imgs_path_fd = open(file_list, "r")
#     imgs_path = imgs_path_fd.readlines()
#     imgs_path_fd.close()
#
#     print(imgs_path)
#
#     str_ret =""
#     for img_path in imgs_path:
#         _str, frame = detect(image_dir, img_path.strip("\n"))
#         str_ret += _str
#         cv2.imwrite(os.path.join(output_dir, img_path.replace("/","_").strip("\n") + ".jpg"), frame)
#
#     d_ret_fd = open(file_result, "w")
#     d_ret_fd.writelines(str_ret)
#     d_ret_fd.close()




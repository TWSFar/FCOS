import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataloaders.transform import IrRegularResizer, Normalizer
from models.retinanet import RetinaNet
from configs.visdrone import opt

import torch

import multiprocessing
multiprocessing.set_start_method('spawn', True)

labels = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'}


def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        id = int(bbox[4])
        label = labels[id]

        if len(bbox) == 6:
            label = label + '{:.2}'.format(bbox[5])

        # plot
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
        c1 = (x1, y1 - t_size[1]-1)
        c2 = (x1 + t_size[0], y1)
        cv2.rectangle(img, c1, c2, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, label, (x1, y1-1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    return img


def test(**kwargs):
    opt._parse(kwargs)
    test_path = "/home/twsf/work/RetinaNet/data/demo"
    imgs_path = os.listdir(test_path)
    resize = IrRegularResizer()
    normalize = Normalizer()

    # Define Network
    # initilize the network here.
    model = RetinaNet(opt, num_classes=80)
    model = model.to(opt.device)

    if os.path.isfile(opt.pre):
        print("=> loading checkpoint '{}'".format(opt.pre))
        checkpoint = torch.load(opt.pre)

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.pre, checkpoint['epoch']))
    else:
        pass
        # raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for img_path in imgs_path:
            # data read and transforms
            img_path = osp.join(test_path, img_path)
            img = cv2.imread(img_path)[:, :, ::-1]
            sample = {'img': img, 'annot': None}
            sample = normalize(sample)
            sample = resize(sample)
            input = sample['img'].unsqueeze(0).to(opt.device).permute(0, 3, 1, 2)

            # predict
            scores, labels, boxes = model(input)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # draw
            boxes = boxes / sample['scale']
            output = torch.cat((boxes, labels.float().unsqueeze(1), scores.unsqueeze(1)), dim=1)
            output = output.cpu().numpy()
            img = draw_bboxes(img, output)

            # show
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 1, 1).imshow(img)
            plt.show()


if __name__ == '__main__':
    test()
    # fire.Fire(test)

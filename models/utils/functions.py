import numpy as np
import torch
import torch.nn as nn
from models.utils.external.nms_gpu import nms, soft_nms
from collections import OrderedDict


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        device = boxes.device
        boxes = boxes.unsqueeze(0)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img_shape):
        height, width = img_shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class PostProcess(object):
    def __init__(self,
                 type="default",
                 pst_thd=0.2,
                 n_pre_nms=4000,
                 nms_thd=0.5):
        self.nms_type = type
        self.pst_thd = pst_thd
        self.n_pre_nms = n_pre_nms
        self.nms_thd = nms_thd
        self.clipBoxes = ClipBoxes()
        self.scr = torch.zeros(0)
        self.lab = torch.zeros(0)
        self.box = torch.zeros(0, 4)

    """ method 1: all category use nms gather
    def __call__(self, scores, labels, boxes):
        scores_list = []
        labels_list = []
        boxes_list = []
        for index in range(len(scores)):
            scores_over_thresh = (scores[index] > self.pst_thd)
            if scores_over_thresh.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scr = scores[index, scores_over_thresh]
            lab = labels[index, scores_over_thresh]
            box = boxes[index, scores_over_thresh]
            bboxes = torch.cat((box, scr.unsqueeze(1)), dim=1)

            nms_idx = nms(bboxes.cpu().numpy(), self.nms_thd)
            scr = scr[nms_idx]
            lab = lab[nms_idx]
            box = box[nms_idx]

            if nms_idx.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(scr.cpu())
            labels_list.append(lab.cpu())
            boxes_list.append(box.cpu())

        return scores_list, labels_list, boxes_list
    """

    # """ method 2: per category use nms along
    def __call__(self, scores_bt, labels_bt, boxes_bt, img_shape):
        boxes_bt = self.clipBoxes(boxes_bt, img_shape)
        scores_list = []
        labels_list = []
        boxes_list = []
        desort_idx = scores_bt.argsort(dim=1, descending=True)
        # olny use the first n which scores are the largest
        desort_idx = desort_idx[:, :self.n_pre_nms]
        for index in range(len(scores_bt)):
            scores = scores_bt[index, desort_idx[index]]
            labels = labels_bt[index, desort_idx[index]]
            boxes = boxes_bt[index, desort_idx[index]]
            scores_over_thresh = (scores > self.pst_thd)
            if scores_over_thresh.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scr = scores[scores_over_thresh]
            lab = labels[scores_over_thresh]
            box = boxes[scores_over_thresh]
            bboxes = torch.cat((box, scr.unsqueeze(1)), dim=1)

            nms_classes = self.lab.type_as(lab)
            nms_scores = self.scr.type_as(scr)
            nms_bboxes = self.box.type_as(box)
            for c in lab.unique():
                idx = lab == c
                b = bboxes[idx]
                c = lab[idx]
                s = scr[idx]
                if self.nms_type == 'soft_nms':
                    nms_idx = soft_nms(b.cpu().numpy(), method=0, threshold=self.pst_thd, Nt=self.nms_thd)
                else:
                    nms_idx = nms(b.cpu().numpy(), self.nms_thd)
                nms_scores = torch.cat((nms_scores, s[nms_idx]), dim=0)
                nms_classes = torch.cat((nms_classes, c[nms_idx]), dim=0)
                nms_bboxes = torch.cat((nms_bboxes, b[nms_idx, :4]), dim=0)

            if len(nms_bboxes) == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(nms_scores.cpu())
            labels_list.append(nms_classes.cpu())
            boxes_list.append(nms_bboxes.cpu())

        return scores_list, labels_list, boxes_list
    # """


def re_resize(pre_bboxes, scale, resize_type):
    """
    args:
        resize_type:
            irregular
            regular
            letterbox
        pre_bboxes:
            tenosr, shape[n, 4]
        scale:
            ....
    """
    # correct boxes for image scale
    if resize_type == "irregular":
        pre_bboxes = pre_bboxes / scale
    elif resize_type == "regular":
        pre_bboxes[:, [0, 2]] = pre_bboxes[:, [0, 2]] / scale[0]
        pre_bboxes[:, [1, 3]] = pre_bboxes[:, [1, 3]] / scale[1]
    elif resize_type == "letterbox":
        pre_bboxes[:, 0] = pre_bboxes[:, 0] / scale[0] - scale[1]
        pre_bboxes[:, 1] = pre_bboxes[:, 1] / scale[0] - scale[2]
        pre_bboxes[:, 2] = pre_bboxes[:, 2] / scale[0] - scale[1]
        pre_bboxes[:, 3] = pre_bboxes[:, 3] / scale[0] - scale[2]

    return pre_bboxes


def iou_cpu(a, b):
    """
    Args:
        a: [N, 4],  4:[x1, y1, x2, y2]
        b: [M, 4],  4:[x1, y1, x2, y2]
    Return:
        IoU: [N, M]
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def nms_cpu(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    order = scores.argsort(descending=True)
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size(0) > 0:
        i = order[0].item()
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        zero = torch.tensor(0.0).to(dets.device)
        w = torch.max(zero, xx2 - xx1)
        h = torch.max(zero, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)

        order = order[1:][iou <= thresh]

    return keep


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    return loss, log_vars

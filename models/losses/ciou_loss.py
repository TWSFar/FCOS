import numpy as np
import torch
import torch.nn as nn
from .utils import (weight_reduce_loss, decode, xyxy_2_xywh,
                    bbox_iou)


def bbox_ciou(box1, box2):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.t().float()
    box2 = box2.t().float()

    # Intersection area
    inter_area = (torch.min(box1[2], box2[2]) - torch.max(box1[0], box2[0])).clamp(0) * \
                 (torch.min(box1[3], box2[3]) - torch.max(box1[1], box2[1])).clamp(0)

    # Union Area
    union_area = ((box1[2] - box1[0]) * (box1[3] - box1[1]) + 1e-16) + \
                 (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area

    iou = inter_area / union_area  # iou

    # Distances both two box center
    box1_cxcy = (box1[2:4] + box1[:2]) / 2
    box2_cxcy = (box2[2:4] + box2[:2]) / 2
    distances = (box1_cxcy - box2_cxcy).pow(2).sum(dim=0)

    # Diagonal of the minimum bounding box
    c_x1, c_x2 = torch.min(box1[0], box2[0]), torch.max(box1[2], box2[2])
    c_y1, c_y2 = torch.min(box1[1], box2[1]), torch.max(box1[3], box2[3])
    diagonal = (c_x1 - c_x2).pow(2) + (c_y1 - c_y2).pow(2)

    box1_wh = box1[2:4] - box1[:2]
    box2_wh = box2[2:4] - box2[:2]
    v = 4 / (np.pi ** 2) * ((box1_wh[0] / box1_wh[1]).atan() - (box2_wh[0] / box2_wh[1]).atan())

    alpha = v / (1 - iou + v)

    ciou = iou - distances / diagonal + alpha * v

    return ciou


class CIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                anchor=None,
                weight=None,
                avg_factor=None):
        """
        Args:
            pred: [N_max, 4],  4: dx, dy, dw, dh
            target: [N_max, 4]  4: x1, y1, x2, y2
            anchor: [N_max, 4]  4: x1, y1, x2 ,y2
        Return:
            loss
        """
        assert pred.shape[0] == target.shape[0]

        variances = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(pred.device)

        if anchor is None:
            pred_box = pred
        else:
            anchor = xyxy_2_xywh(anchor)
            pred_box = decode(pred, anchor, variances)

        ious = bbox_iou(pred_box, target).clamp(min=self.eps)
        loss = -ious.log()

        loss = self.loss_weight * weight_reduce_loss(
            loss,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        return loss

if __name__ == "__main__":
    m1 = torch.tensor([[0., 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4]])
    m2 = torch.tensor([[0., 0, 5, 5], [2, 2, 3, 3], [0, 0, 2, 4]])
    ciou = bbox_ciou(m1, m2)
    pass

import torch


def xywh_2_xyxy(boxes):
    """
    Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


def xyxy_2_xywh(boxes):
    """
    Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,
                      boxes[:, 2:] - boxes[:, :2]), dim=1)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def encode(matched, priors, variances):
    """
    Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].  4: [x1, y1, x2, y2]
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].  4: [x, y, w, h]
        variances: (tensor) Variances of priorboxes, [vx, vy, vw, vh]
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy = g_cxcy / priors[:, 2:]

    # match wh / prior wh
    g_wh = torch.clamp(matched[:, 2:] - matched[:, :2], min=1.)
    g_wh = torch.log(g_wh / priors[:, 2:])

    return torch.cat([g_cxcy, g_wh], 1) / variances


def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4].  4: [dx, dy, dw, dh]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].  4: [x, y, w, h]
        variances: (tensor) Variances of priorboxes, [vx, vy, vw, vh]
    Return:
        decoded bounding box predictions
    """
    loc = loc * variances
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:])), 1)

    boxes = xywh_2_xyxy(boxes)

    return boxes  # xyxy


def bbox_iou(box1, box2, GIoU=False):
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

    if GIoU:
        c_x1, c_x2 = torch.min(box1[0], box2[0]), torch.max(box1[2], box2[2])
        c_y1, c_y2 = torch.min(box1[1], box2[1]), torch.max(box1[3], box2[3])
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area

        iou = iou - (c_area - union_area) / c_area  # GIoU

    return iou


if __name__ == "__main__":
    a = torch.rand(3, 4)
    b = torch.rand(3, 4)
    c = torch.rand(1, 4)
    pred = decode(a, b, c)
    temp = encode(pred, b, c)
    a = torch.tensor([[0, 0, 4, 4], [0, 0, 4, 4]])
    b = torch.tensor([[0, 0, 4, 4], [1, 1, 2, 2]])
    iou = bbox_iou(a, b)
    giou = bbox_iou(a, b, GIoU=True)
    pass

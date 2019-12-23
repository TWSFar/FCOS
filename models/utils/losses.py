import torch


def bbox_overlaps_aligned(bboxes1, bboxes2, is_aligned=False):
    '''
    Param:
    bboxes1:   FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
    bboxes2:   FloatTensor(n, 4)

    Return:
    FloatTensor(n)
    '''
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)
    return ious


class Losses(object):
    def __init__(self, cls_loss="focalloss", reg_loss="iou", gamma=2.0, alpha=0.25):
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, cls_out, reg_out, targets_cls, targets_reg):
        mask_pos = targets_cls > -1
        num_pos = torch.sum(mask_pos, dim=1).clamp_(min=1)  # (b)
        batch_size = targets_cls.shape[0]
        loss_cls = []
        loss_reg = []
        for b in range(batch_size):
            cls_out_b = cls_out[b]
            targets_cls_b = targets_cls[b]
            if self.cls_loss == "focalloss":
                loss_cls_b = self.focalloss(cls_out_b, targets_cls_b)
            loss_cls.append(loss_cls_b.sum().view(1) / num_pos[b])

            reg_out_b = reg_out[b][mask_pos[b]]
            targets_reg_b = targets_reg[b][mask_pos[b]]
            if self.reg_loss == "iou":
                loss_reg_b = self.iou_loss(reg_out_b, targets_reg_b)
            loss_reg.append(loss_reg_b.sum().view(1) / num_pos[b])

        return torch.cat(loss_cls, dim=0), torch.cat(loss_reg, dim=0)

    def focalloss(self, input, target):
        target = target.long() + 1
        one_hot = torch.zeros(input.shape[0],
                              1 + input.shape[1]).to(input.device).scatter_(
                                  1, target.view(-1, 1), 1)
        one_hot = one_hot[:, 1:]
        pt = input * one_hot + (1.0-input) * (1.0-one_hot)
        w = self.alpha * one_hot + (1.0 - self.alpha) * (1.0-one_hot)
        w = w * torch.pow((1.0-pt), self.gamma)
        loss = -w * pt.log()
        return loss

    def iou_loss(self, preds, targets, eps=1e-6):
        '''
        Param:
            pred:     FloatTensor(n, 4)
            target:   FloatTensor(n, 4)

        Return:
            FloatTensor(n)
        '''
        ious = bbox_overlaps_aligned(preds, targets).clamp(min=eps)
        loss = -ious.log()
        return loss

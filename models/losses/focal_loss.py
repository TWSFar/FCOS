import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss


class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):
        device = pred.device
        assert pred.shape[0] == target.shape[0]

        if len(target.shape) == 1:
            target = torch.zeros(
                pred.shape,
                dtype=torch.long,
                device=device).scatter_(
                    1, target.view(-1, 1), 1)

        pred_sigmoid = pred.sigmoid()
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-4, 1.0 - 1e-4)
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight

        loss = torch.where(torch.ne(target, -1.0),
                           loss, torch.zeros(loss.shape).to(device))

        loss = self.loss_weight * weight_reduce_loss(
            loss,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        return loss

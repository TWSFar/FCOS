import torch
import torch.nn as nn
from models.utils import iou_cpu


class FocalLoss(nn.Module):
    def __init__(self, giou_loss=False, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        self.giou_loss = giou_loss
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        device = regressions.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :].sigmoid()
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(device))
                classification_losses.append(torch.tensor(0).float().to(device))

                continue

            # num_anchors x num_annotations
            IoU = iou_cpu(anchors, bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(device)

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).to(device) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1.-alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1.-classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                # negative_indices = 1 - positive_indices
                predicts = regression[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

                regression_diff = torch.abs(targets - predicts)

                # Smooth L1 Loss
                regression_loss = torch.where(
                    torch.le(regression_diff, 0.11),
                    0.5 * torch.pow(regression_diff, 2) / 0.11,
                    regression_diff - 0.5 * 0.11
                )

                regression_losses.append(regression_loss.mean())

            else:
                regression_losses.append(torch.tensor(0).float().to(device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True)

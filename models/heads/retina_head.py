import math
import torch
import torch.nn as nn
from ..losses import build_loss
from ..utils import (Anchors, BBoxTransform, iou_cpu)

# from models.losses.debug import FocalLoss


class RetinaHead(nn.Module):
    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 in_channels=256,
                 feat_channels=256,
                 nclass=80):
        super(RetinaHead, self).__init__()
        self.nclass = nclass
        self.anchors = Anchors(strides=strides)
        self.nanchor = len(self.anchors)
        self.regressBoxes = BBoxTransform()

        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU())
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU())

        self.retina_cls = nn.Conv2d(
            feat_channels, self.nanchor*self.nclass, kernel_size=3, padding=1)
        self.retina_reg = nn.Conv2d(
            feat_channels, self.nanchor*4, kernel_size=3, padding=1)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        prior = 0.01
        for conv in [self.retina_cls, self.retina_reg]:
            conv.weight.data.fill_(0)
            conv.bias.data.fill_(-math.log((1.0-prior)/prior))

    def forward(self, inputs, annotations=None):
        device = inputs[0].device
        num_imgs = inputs[0].size(0)

        cls_feats = [self.cls_convs(featmap) for featmap in inputs]
        cls_scores = [self.retina_cls(featmap) for featmap in cls_feats]
        reg_feats = [self.reg_convs(featmap) for featmap in inputs]
        reg_preds = [self.retina_reg(featmap) for featmap in reg_feats]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchors, resticts = self.anchors(
            featmap_sizes, dtype=reg_preds[0].dtype, device=device)

        batchs_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.nclass*self.nanchor)
            for cls_score in cls_scores], dim=1)
        batchs_reg_preds = torch.cat([
            reg_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*self.nanchor)
            for reg_pred in reg_preds], dim=1)

        if self.training:
            # return self.loss(pred_cls, pred_reg, anchors[0], annotations)
            labels_list, bbox_targets_list = self.retina_target(
                annotations, anchors, resticts, device)
            batchs_labels = torch.cat(labels_list)
            pos_inds = (batchs_labels > 0).max(dim=1)[0]
            batchs_bbox_targets = torch.cat(bbox_targets_list)
            batchs_anchors = anchors.repeat(num_imgs, 1)

            num_pos = pos_inds.sum()

            loss_cls = self.loss_cls(
                batchs_cls_scores.reshape(-1, self.nclass),
                batchs_labels,
                avg_factor=num_pos)

            if num_pos > 0:
                loss_bbox = self.loss_bbox(
                    batchs_reg_preds.reshape(-1, 4)[pos_inds],
                    batchs_bbox_targets[pos_inds],
                    batchs_anchors[pos_inds])
            else:
                loss_bbox = torch.tensor(0).float().to(device)

            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox
            )

        else:
            batchs_bbox_preds = self.regressBoxes(
                anchors, batchs_reg_preds.reshape(num_imgs, -1, 4))

            scores, classes = torch.max(
                batchs_cls_scores.sigmoid(), dim=2, keepdim=True)

            return scores.squeeze(2), classes.squeeze(2), batchs_bbox_preds

    def retina_target(self, annotations, anchors, resticts, deivce):
        labels_list, bbox_targets_list = [], []
        for annotation in annotations:
            labels, bbox_targets = self.retina_single_target(
                annotation, anchors, resticts)
            labels_list.append(labels.to(deivce))
            bbox_targets_list.append(bbox_targets)

        return labels_list, bbox_targets_list

    def retina_single_target(self, annotation, anchors, resticts):
        targets = torch.ones(anchors.shape[0],
                             self.nclass,
                             dtype=torch.long) * -1

        # num_anchors x num_annotations
        IoU = iou_cpu(anchors, annotation[:, :4])
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

        assigned_annotations = annotation[IoU_argmax, :]
        # agd_ann_wh = assigned_annotations[:, 2:4] - assigned_annotations[:, :2]
        # scale_indices = (agd_ann_wh.max(dim=1)[0] > resticts[:, 0]) \
        #     & (agd_ann_wh.max(dim=1)[0] < resticts[:, 1])

        positive_indices = torch.ge(IoU_max, 0.5)

        targets[torch.lt(IoU_max, 0.4), :] = 0
        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

        return targets, assigned_annotations[:, :4]

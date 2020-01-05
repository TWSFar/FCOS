import math
import torch
import torch.nn as nn
from ..losses import build_loss
from ..utils import (Anchors, BBoxTransform, iou_cpu)

# from models.losses.debug import FocalLoss


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0-prior)/prior))

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0-prior)/prior))

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


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

        self.retina_cls = ClassificationModel(num_features_in=256,
                                              num_anchors=self.nanchor,
                                              num_classes=self.nclass)
        self.retina_reg = RegressionModel(num_features_in=256, num_anchors=9)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def forward(self, inputs, annotations=None):
        device = inputs[0].device
        num_imgs = inputs[0].size(0)
        tensor_zero = torch.tensor(0).float().to(device)
        featmap_sizes = [featmap.size()[-2:] for featmap in inputs]
        anchors, resticts = self.anchors(
            featmap_sizes, dtype=inputs[0].dtype, device=device)

        pred_cls = torch.cat(
            [self.retina_cls(featmap) for featmap in inputs], dim=1)
        pred_reg = torch.cat(
            [self.retina_reg(featmap) for featmap in inputs], dim=1)

        if self.training:
            # return self.loss(pred_cls, pred_reg, anchors, annotations)
            loss_cls, loss_reg = [], []

            for bi in range(num_imgs):
                annotation = annotations[bi]
                annotation = annotation[annotation[:, 4] != -1]
                if annotation.shape[0] == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                target_cls, target_bbox, pst_idx = self._encode(anchors,
                                                                annotation,
                                                                resticts[0])
                if pst_idx.sum() == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                loss_cls_bi = self.loss_cls(pred_cls[bi],
                                            target_cls,
                                            avg_factor=pst_idx.sum())
                loss_reg_bi = self.loss_bbox(pred_reg[bi, pst_idx],
                                             target_bbox,
                                             anchors[pst_idx])
                loss_cls.append(loss_cls_bi)
                loss_reg.append(loss_reg_bi)

            return dict(
                loss_cls=torch.stack(loss_cls).mean(),
                loss_reg=torch.stack(loss_reg).mean()
            )

        else:
            transformed_anchors = self.regressBoxes(
                anchors, pred_reg)

            scores, class_id = torch.max(
                pred_cls.sigmoid(), dim=2, keepdim=True)

            return scores.squeeze(2), class_id.squeeze(2), transformed_anchors

    def _encode(self, anchors, annotation, resticts):
        device = anchors.device
        targets = torch.ones(anchors.shape[0], self.nclass) * -1
        targets = targets.to(device)

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
        targets[positive_indices,
                assigned_annotations[positive_indices, 4].long()] = 1

        return targets, \
            assigned_annotations[positive_indices, :4], \
            positive_indices

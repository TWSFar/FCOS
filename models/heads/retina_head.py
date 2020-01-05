import math
import torch
import torch.nn as nn
from ..losses import build_loss
from ..utils import (Anchors, BBoxTransform,
                     ClipBoxes, iou_cpu)

INF = 1e8

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
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

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

        self.retina_cls = nn.Conv2d(feat_channels, 9*4, kernel_size=3, padding=1)
        self.retina_reg = nn.Conv2d(feat_channels, 9*self.nclass, kernel_size=3, padding=1)

        self.cls_loss = build_loss(loss_cls)
        self.reg_loss = build_loss(loss_bbox)

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
        tensor_zero = torch.tensor(0).float().to(device)

        cls_feats = [self.cls_convs(featmap) for featmap in inputs]
        cls_scores = [self.retina_cls(featmap) for featmap in cls_feats]
        reg_feats = [self.reg_convs(featmap) for featmap in inputs]
        bbox_preds = [self.retina_reg(featmap) for featmap in reg_feats]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_anchors, all_level_resticts = self.anchors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=device)

        if self.training:
            # return self.loss(pred_cls, pred_reg, anchors[0], annotations)
            loss_cls, loss_reg = [], []

            for bi in range(num_imgs):
                annotation = annotations[bi]
                annotation = annotation[annotation[:, 4] != -1]
                if annotation.shape[0] == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                target_cls, target_bbox, pst_idx = self._encode(anchors[0],
                                                                annotation,
                                                                resticts[0])
                if pst_idx.sum() == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                loss_cls_bi = self.cls_loss(pred_cls[bi], target_cls)
                loss_reg_bi = self.reg_loss(pred_reg[bi, pst_idx],
                                            target_bbox,
                                            anchors[0, pst_idx])
                loss_cls.append(loss_cls_bi.sum()/torch.clamp(pst_idx.sum().float(), min=1.0))
                loss_reg.append(loss_reg_bi.mean())

            return torch.stack(loss_cls).mean(dim=0, keepdim=True), \
                torch.stack(loss_reg).mean(dim=0, keepdim=True)

        else:
            transformed_anchors = self.regressBoxes(anchors, pred_reg)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores, class_id = torch.max(pred_cls.sigmoid(), dim=2, keepdim=True)

            return scores.squeeze(2), class_id.squeeze(2), transformed_anchors

    def _encode(self, anchors, annotation, resticts):
        device = anchors.device
        targets = torch.ones(anchors.shape[0], self.num_classes) * -1
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
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

        return targets, assigned_annotations[positive_indices, :4], positive_indices


if __name__ == "__main__":
    from configs.visdrone_chip import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out3 = model(input)
    pass



if __name__ == "__main__":
    from configs.visdrone_chip import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out3 = model(input)
    pass

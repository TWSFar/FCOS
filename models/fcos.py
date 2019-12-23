import torch
import torch.nn as nn

from models import backbones
from models.heads.classification import ClassificationModel
from models.heads.regression import RegressionModel
from models.utils import losses, anchors


class FCOS(nn.Module):

    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(FCOS, self).__init__()
        self.backbone = backbones.build_backbone(opt)

        self.reg_model = RegressionModel(num_features_in=256,
                                         num_feaures_layer=5)
        self.cls_model = ClassificationModel(num_features_in=256,
                                             num_classes=num_classes)
        self.anchor = anchors.Anchors()

        self.losses = losses.Losses()

        self.freeze_bn()

    def freeze_bn(self):
        """Freeeze BarchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        features = self.backbone(img_batch)
        reg_out = torch.cat([self.reg_model(feature, i) for i, feature in enumerate(features)], dim=1)
        cls_out = torch.cat([self.cls_model(feature) for feature in features], dim=1)
        # generate anchors
        _center_xy, _tlbr_max_minmax, _center_offset_max = self.anchor(img_batch)

        if self.training:
            targets_cls, targets_reg = self._encode(annotations,
                                                    _center_xy,
                                                    _tlbr_max_minmax,
                                                    _center_offset_max)
            for b in range(targets_cls.shape[0]):
                reg_out[b] = self.distance2bbox(_center_xy, reg_out[b])
                targets_reg[b] = self.distance2bbox(_center_xy, targets_reg[b])

            return self.losses(cls_out, reg_out, targets_cls, targets_reg)
        else:
            score_preds, class_preds = torch.max(cls_out, dim=2)

            reg_preds = [self.distance2bbox(_center_xy, reg_tlbr) for reg_tlbr in reg_out]
            reg_preds = torch.stack(reg_preds)  # (b, shw)

            return score_preds, class_preds, reg_preds

    def _encode(self, annotations,
                _center_xy, _tlbr_max_minmax, _center_offset_max):
        '''
        Param:
            annotations: LongTensor(batch_num, N_max, 5)  5: xyxy label
            _center_xy: FloatTensor((x1, y1), (x2, y1)...(xn, yn))
            _tlbr_max_minmax: FloatTensor((low_1, high_1), .., (low_n, high_n))
            _center_offset_max: FloatTensor(12)

        Return:
            targets_cls: LongTensor(batch_num, shw)
            targets_reg: FloatTensor(batch_num, shw, 4)
        '''
        targets_cls, targets_reg = [], []
        for b in range(annotations.shape[0]):
            # get tlbr
            mask_target = annotations[b, :, 4] != -1
            label_box = annotations[b, mask_target, :4]
            label_class = annotations[b, mask_target, 4]
            tl = _center_xy[:, None, :] - label_box[:, :2]
            br = label_box[:, 2:] - _center_xy[:, None, :]
            tlbr = torch.cat([tl, br], dim=2)

            # get center_offset
            label_xy = (label_box[:, :2] + label_box[:, 2:]) / 2.0
            center_offset_xy = _center_xy[:, None, :] - label_xy[:, :]
            center_offset = (center_offset_xy[:, :, 0]**2 +
                             center_offset_xy[:, :, 1]**2).sqrt()

            # get mask_pos
            _min = torch.min(tlbr, dim=2)[0]
            _max = torch.max(tlbr, dim=2)[0]
            mask_inside = _min > 0
            mask_scale = (_max > _tlbr_max_minmax[:, None, 0]) \
                & (_max <= _tlbr_max_minmax[:, None, 1])
            mask_center_offset = center_offset < \
                _center_offset_max.view(-1, 1).expand_as(center_offset)
            mask_pos = mask_inside & mask_scale & mask_center_offset

            # get area
            area = (tlbr[:, :, 0] + tlbr[:, :, 2]) \
                * (tlbr[:, :, 1] + tlbr[:, :, 3])
            area[~mask_pos] = 999999999  # inf

            # get area_min_index
            area_min_index = torch.min(area, dim=1)[1]

            # get targets_cls_b
            targets_cls_b = label_class[area_min_index]
            mask_pos_shw = torch.max(mask_pos, dim=1)[0]
            targets_cls_b[~mask_pos_shw] = -1  # background: -1

            # get targets_reg_b
            targets_reg_b = tlbr[
                torch.zeros_like(area,
                                 dtype=torch.bool).scatter_(
                    1, area_min_index.view(-1, 1), 1)]

            # # ignore
            # cd1 = _center_xy - loc[b, :2]
            # cd2 = loc[b, 2:] - _center_xy
            # mask = (cd1.min(dim=1)[0] < 0) | (cd2.min(dim=1)[0] < 0)
            # targets_cls_b[mask] = -1

            # append
            targets_cls.append(targets_cls_b)
            targets_reg.append(targets_reg_b)

        return torch.stack(targets_cls), torch.stack(targets_reg)

    def distance2bbox(self, points, distance):
        '''
        Param:
        points:   FloatTensor(n, 2)  2: x y
        distance: FloatTensor(n, 4)  4: top left bottom right

        Return:
        FloatTensor(n, 4) 4: ymin xmin ymax xmax
        '''
        xmin = points[:, 0] - distance[:, 1]
        ymin = points[:, 1] - distance[:, 0]
        xmax = points[:, 0] + distance[:, 3]
        ymax = points[:, 1] + distance[:, 2]
        return torch.stack([xmin, ymin, xmax, ymax], -1)


if __name__ == "__main__":
    from configs.fcos_res50_coco import opt
    model = FCOS(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out2 = model(input)
    pass

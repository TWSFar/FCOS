import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..losses import build_loss
from ..utils import Scale, bias_init_with_prob

INF = 1e8


class FCOSHead(nn.Module):

    def __init__(self,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 strides=[],
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                 in_channels=256,
                 feat_channels=256,
                 nclass=80):
        super(FCOSHead, self).__init__()
        self.nclass = nclass
        self.strides = strides
        self.regress_ranges = regress_ranges

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
        self.fcos_cls = nn.Conv2d(feat_channels, self.nclass, 3, padding=1)
        self.fcos_reg = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, inputs, annotations=None):
        import time
        time1 = time.time()
        cls_feats = [self.cls_convs(featmap) for featmap in inputs]
        cls_scores = [self.fcos_cls(featmap) for featmap in cls_feats]
        centernesses = [self.fcos_centerness(featmap) for featmap in cls_feats]
        reg_feats = [self.reg_convs(featmap) for featmap in inputs]
        bbox_preds = [self.scales[i](self.fcos_reg(featmap)).exp()
                      for i, featmap in enumerate(reg_feats)]

        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes,
                                           bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        if self.training:
            labels, bbox_targets = self.fcos_target(all_level_points,
                                                    annotations)

            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(-1, self.nclass)
                for cls_score in cls_scores
            ]
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                for bbox_pred in bbox_preds
            ]
            flatten_centerness = [
                centerness.permute(0, 2, 3, 1).reshape(-1)
                for centerness in centernesses
            ]
            flatten_cls_scores = torch.cat(flatten_cls_scores)
            flatten_bbox_preds = torch.cat(flatten_bbox_preds)
            flatten_centerness = torch.cat(flatten_centerness)
            flatten_labels = torch.cat(labels)
            flatten_bbox_targets = torch.cat(bbox_targets)
            flatten_points = torch.cat(
                [points.repeat(num_imgs, 1) for points in all_level_points])

            pos_inds = flatten_labels.nonzero().reshape(-1)
            num_pos = len(pos_inds)
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels,
                avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_centerness = flatten_centerness[pos_inds]

            if num_pos > 0:
                pos_bbox_targets = flatten_bbox_targets[pos_inds]
                pos_centerness_targets = self.centerness_target(
                    pos_bbox_targets)
                pos_points = flatten_points[pos_inds]
                pos_decoded_bbox_preds = self.distance2bbox(pos_points,
                                                            pos_bbox_preds)
                pos_decoded_target_preds = self.distance2bbox(pos_points,
                                                              pos_bbox_targets)
                # centerness weighted iou loss
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())
                loss_centerness = self.loss_centerness(pos_centerness,
                                                       pos_centerness_targets)
            else:
                loss_bbox = pos_bbox_preds.sum()
                loss_centerness = pos_centerness.sum()

            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)

        else:
            time2 = time.time()
            flatten_cls_scores = torch.cat([
                cls_score.permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, self.nclass).sigmoid()
                for cls_score in cls_scores], dim=1)
            flatten_centerness = torch.cat([
                centerness.permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, 1).sigmoid()
                for centerness in centernesses], dim=1)
            flatten_bbox_preds = torch.cat([
                bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                for bbox_pred in bbox_preds], dim=1)
            score_preds, class_preds = torch.max(
                flatten_cls_scores * flatten_centerness, dim=2)
            reg_preds = [
                self.distance2bbox(
                    torch.cat(all_level_points), flatten_bbox_pred)
                for flatten_bbox_pred in flatten_bbox_preds
            ]
            reg_preds = torch.stack(reg_preds)  # (b, shw)
            time3 = time.time()
            print("head 1-2: {}".format(time2-time1))
            print("head 2-3: {}".format(time3-time2))
            return score_preds, class_preds, reg_preds

    def distance2bbox(self, points, distance):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        return torch.stack([x1, y1, x2, y2], -1)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, annotations):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list = []
        bbox_targets_list = []
        for annotation in annotations:
            gt_index = annotation[:, 4] != -1
            gt_bboxes = annotation[gt_index, :4]
            gt_labels = annotation[gt_index, 4].long()
            labels, bbox_targets = \
                self.fcos_target_single(gt_bboxes, gt_labels,
                                        concat_points, concat_regress_ranges)
            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


if __name__ == "__main__":
    from configs.fcos_res50_coco import opt
    model = FCOSHead(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out2 = model(input)
    pass

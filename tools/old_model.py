import math
import torch
import torch.nn as nn

from models.classification import ClassificationModel
from models.regression import RegressionModel
from models.backbones.resnet import resnet50, resnet101
from models.utils import losses
from models.utils.anchors import Anchors
from models.utils.functions import BBoxTransform, ClipBoxes
from models.utils.nms.nms_gpu import nms


class RetinaNet(nn.Module):

    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(RetinaNet, self).__init__()
        self.pst_thd = self.opt.pst_thd  # positive threshold
        self.nms_thd = self.opt.nms_thd
        self.backbone = resnet50()
        # self.backbone = resnet101()
        self.regressionModel = RegressionModel(num_features_in=256)
        self.classificationModel = ClassificationModel(num_features_in=256,
                                                       num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

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

        device = img_batch.device

        features = self.backbone(img_batch)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch).to(regression.device)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores, class_id = torch.max(classification, dim=2, keepdim=True)

            scores_over_thresh = (scores > self.pst_thd)[0, :, 0]

            nms_scores = torch.zeros(0).to(device)
            nms_class = torch.zeros(0).to(device)
            nms_bbox = torch.zeros(0, 4).to(device)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return nms_scores, nms_class, nms_bbox

            # classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            class_id = class_id[:, scores_over_thresh, :]

            bbox = torch.cat([transformed_anchors, scores], dim=2)[0, :, :]
            scores = scores[0].squeeze(1)
            class_id = class_id[0].squeeze(1)

            """ method 1: every category along use nms
            nms_classes = nms_class.type_as(class_id)
            nms_scores = nms_scores.type_as(scores)
            nms_bboxes = nms_bbox.type_as(bbox)
            for c in class_id.unique():
                idx = class_id == c
                b = bbox[idx]
                c = class_id[idx]
                s = scores[idx]
                nms_idx = nms(b.cpu().numpy(), self.nms_thd)
                nms_scores = torch.cat((nms_scores, s[nms_idx]), dim=0)
                nms_classes = torch.cat((nms_classes, c[nms_idx]), dim=0)
                nms_bboxes = torch.cat((nms_bboxes, b[nms_idx, :4]), dim=0)
            """

            # method 2: all category gather use nms
            nms_idx = nms(bbox.cpu().numpy(), self.nms_thd)
            nms_scores = scores[nms_idx]
            nms_classes = class_id[nms_idx]
            nms_bboxes = bbox[nms_idx, :4]

            return nms_scores, nms_classes, nms_bboxes


if __name__ == "__main__":
    from utils.config import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out2 = model(input)
    pass

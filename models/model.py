import torch
import torch.nn as nn

from models import backbones, necks, heads


class Model(nn.Module):

    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(Model, self).__init__()
        self.backbone = backbones.build_backbone(opt.backbone)
        self.neck = necks.build_neck(neck=opt.neck,
                                     in_planes=self.backbone.out_planes,
                                     out_plane=256)
        opt.head["nclass"] = num_classes
        self.head = heads.build_head(opt.head)

        if opt.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        """Freeeze BarchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, imgs, targets=None):
        featmaps = self.backbone(imgs)
        featmaps = self.neck(featmaps)
        if self.training:
            return self.head(featmaps, targets)
        else:
            return self.head(featmaps)


if __name__ == "__main__":
    from configs.fcos_res50_coco import opt
    model = Model(opt)
    inputs = torch.rand(2, 3, 32, 32)
    pass

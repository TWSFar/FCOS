from .hrnet import hrnet
from .resnet import resnet50, resnet101


def build_backbone(backbone):
    if backbone == 'resnet101':
        return resnet101()

    elif backbone == 'resnet50':
        return resnet50()

    elif 'hrnet' in backbone:
        return hrnet(backbone)

    else:
        raise NotImplementedError

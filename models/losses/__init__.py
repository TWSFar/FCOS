from .focal_loss import FocalLoss
from .smooth_l1_loss import SmoothL1Loss
from .iou_loss import IoULoss
from .giou_loss import GIoULoss
from .balanced_l1_loss import BalancedL1Loss
from .ciou_loss import CIoULoss
from .cross_entropy_loss import CrossEntropyLoss


def build_loss(args):
    obj_type = args.pop('type')

    if obj_type == 'FocalLoss':
        return FocalLoss(**args)

    elif obj_type == "SmoothL1Loss":
        return SmoothL1Loss(**args)

    elif obj_type == "BalancedL1Loss":
        return BalancedL1Loss(**args)

    elif obj_type == "GIoULoss":
        return GIoULoss(**args)

    elif obj_type == "IoULoss":
        return IoULoss(**args)

    elif obj_type == "CIoULoss":
        return CIoULoss(**args)

    elif obj_type == "CrossEntropyLoss":
        return CrossEntropyLoss(**args)

    else:
        raise NotImplementedError

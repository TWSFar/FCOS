from .fpn import FPN
from .fpn_aug_se import FPN_AUG_SE
from .fpn_se import FPN_SE
from .fpn_gate import FPN_GATE
from .hrnet_neck import HRNET_NECK


def build_neck(neck, in_planes, out_plane=256):
    if neck == 'fpn':
        assert len(in_planes) == 3
        return FPN(in_planes, out_plane)
    elif neck == 'fpn_aug_se':
        assert len(in_planes) == 3
        return FPN_AUG_SE(in_planes, out_plane)
    elif neck == 'fpn_se':
        assert len(in_planes) == 3
        return FPN_SE(in_planes, out_plane)
    elif neck == 'fpn_gate':
        assert len(in_planes) == 3
        return FPN_GATE(in_planes, out_plane)
    elif neck == 'hrnet_neck':
        assert len(in_planes) == 4
        return HRNET_NECK(in_planes, out_plane)
    else:
        raise NotImplementedError

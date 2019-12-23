import numpy as np
import torch
import torch.nn as nn


def gen_anchors(image_shape,
                strides=[8, 16, 32, 64, 128],
                pyramid_levels=[3, 4, 5, 6, 7],
                tlbr_max_regions=[0, 64, 128, 256, 512, 99999],
                center_offset_ratio=1.5):

    if isinstance(image_shape, int):
        image_shape = np.array((image_shape, image_shape))
    else:
        image_shape = np.array(image_shape)

    center_xy, tlbr_max_minmax, center_offset_max = [], [], []
    feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    for idx, p in enumerate(pyramid_levels):
        fsz_h = feature_shapes[idx][0]
        fsz_w = feature_shapes[idx][1]
        center_x = (np.arange(0, fsz_w) + 0.5) * strides[idx]
        center_y = (np.arange(0, fsz_h) + 0.5) * strides[idx]
        center_x, center_y = np.meshgrid(center_x, center_y)

        center_xy_i = np.vstack((
            center_x.ravel(), center_y.ravel(),
        )).transpose().astype(np.float32)
        center_xy.append(torch.tensor(center_xy_i))

        tlbr_max_minmax_i = torch.zeros(fsz_h*fsz_w, 2)
        tlbr_max_minmax_i[:, 0] = tlbr_max_regions[idx]
        tlbr_max_minmax_i[:, 1] = tlbr_max_regions[idx + 1]
        tlbr_max_minmax.append(tlbr_max_minmax_i)

        center_offset_max_i = torch.zeros(fsz_h*fsz_w)
        center_offset_max_i[:] = float(strides[idx]) * center_offset_ratio
        center_offset_max.append(center_offset_max_i)
    return torch.cat(center_xy, dim=0), \
        torch.cat(tlbr_max_minmax, dim=0), \
        torch.cat(center_offset_max, dim=0)


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7], strides=None,
                 tlbr_max_regions=[0, 64, 128, 256, 512, 99999],
                 center_offset_ratio=1.5):
        self.pyramid_levels = pyramid_levels
        self.tlbr_max_regions = tlbr_max_regions
        self.center_offset_ratio = center_offset_ratio
        super(Anchors, self).__init__()

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides

    def __call__(self, image):
        """
        Args:
            features: (batch, channel, h, w)
        """
        device = image.device
        image_shape = np.array(image.shape[2:])

        center_xy, tlbr_max_minmax, center_offset_max = [], [], []
        feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        for idx, p in enumerate(self.pyramid_levels):
            fsz_h = feature_shapes[idx][0]
            fsz_w = feature_shapes[idx][1]
            center_x = (np.arange(0, fsz_w) + 0.5) * self.strides[idx]
            center_y = (np.arange(0, fsz_h) + 0.5) * self.strides[idx]
            center_x, center_y = np.meshgrid(center_x, center_y)

            center_xy_i = np.vstack((
                center_x.ravel(), center_y.ravel(),
            )).transpose().astype(np.float32)
            center_xy.append(torch.tensor(center_xy_i))

            tlbr_max_minmax_i = torch.zeros(fsz_h*fsz_w, 2)
            tlbr_max_minmax_i[:, 0] = self.tlbr_max_regions[idx]
            tlbr_max_minmax_i[:, 1] = self.tlbr_max_regions[idx + 1]
            tlbr_max_minmax.append(tlbr_max_minmax_i)

            center_offset_max_i = torch.zeros(fsz_h*fsz_w)
            center_offset_max_i[:] = float(self.strides[idx]) * self.center_offset_ratio
            center_offset_max.append(center_offset_max_i)

        return torch.cat(center_xy, dim=0).to(device), \
            torch.cat(tlbr_max_minmax, dim=0).to(device), \
            torch.cat(center_offset_max, dim=0).to(device)


if __name__ == "__main__":
    import numpy as np
    temp1 = gen_anchors(np.array([43, 85]))
    anchor = Anchors()
    
    temp2 = anchor(torch.rand(3, 3, 43, 85))
    pass

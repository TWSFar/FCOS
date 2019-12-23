import numpy as np
import torch
import torch.nn as nn


def gen_anchors(img_size, strides,
                tlbr_max_regions, center_offset_ratio=1.5):
    '''
    Return:
    center_yx:            FloatTensor(acc_scale(Hi*Wi), 2)
    tlbr_max_minmax:      FloatTensor(acc_scale(Hi*Wi), 2)
    center_offset_max:    FloatTensor(acc_scale(Hi*Wi))

    Note:
    - Hi,Wi accumulate from big to small

    Example:
    img_size = (h:608, w:1025)
    strides = [8, 16, 32, 64, 128]  # the stride of smallest anchors
    tlbr_max_regions = [0, 64, 128, 256, 512, 9999] # 5 scales
    center_offset_ratio = 1.5
    '''
    center_xy, tlbr_max_minmax, center_offset_max = [], [], []
    for id, stride in enumerate(strides):
        fsz_h = (img_size[0]-1) // stride + 1
        fsz_w = (img_size[1]-1) // stride + 1
        center_xy_i = torch.zeros(fsz_h, fsz_w, 2)
        for w in range(fsz_w):
            for h in range(fsz_h):
                c_y, c_x = h * float(stride), w * float(stride)
                center_xy_i[h, w, :] = torch.Tensor([c_x, c_y])
        center_xy_i = center_xy_i.view(fsz_h*fsz_w, 2)
        tlbr_max_minmax_i = torch.zeros(fsz_h*fsz_w, 2)
        tlbr_max_minmax_i[:, 0] = tlbr_max_regions[id]
        tlbr_max_minmax_i[:, 1] = tlbr_max_regions[id + 1]
        center_offset_max_i = torch.zeros(fsz_h*fsz_w)
        center_offset_max_i[:] = float(stride) * center_offset_ratio
        center_xy.append(center_xy_i)
        tlbr_max_minmax.append(tlbr_max_minmax_i)
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
    temp1 = gen_anchors((43, 85), [8, 16, 32, 64, 128], [0, 64, 128, 256, 512, 9999])
    anchor = Anchors()
    # temp2 = anchor((43, 85))
    pass

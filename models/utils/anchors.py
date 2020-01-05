import numpy as np
import torch
import torch.nn as nn


class Anchors(object):
    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7],
                 strides=None, sizes=None, ratios=None, scales=None,
                 gt_restrict_range=[0, 64, 128, 256, 512, 99999]):
        super(Anchors, self).__init__()

        assert len(pyramid_levels) == len(gt_restrict_range) - 1

        self.pyramid_levels = pyramid_levels
        self.gt_restrict_range = gt_restrict_range

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides

        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        else:
            self.sizes = sizes

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0/3.0), 2 ** (2.0/3.0)])
        else:
            self.scales = scales

        self.anchors = []
        for size in self.sizes:
            self.anchors.append(self.generate_anchors(base_size=size))

    def __call__(self, featmap_sizes, dtype, device):
        """
        Args:
            featmap_sizes:
                featmap sizes, according to pyramid levels order
        Return:
            anchors: [1, num, 4]  4: [x1, y1, x2, y2]
        """
        assert len(featmap_sizes) == len(self.pyramid_levels)

        # compute anchors over all pyramid levels
        # all_anchors = np.zeros((0, 4)).astype(np.float32)
        # all_restrictions = np.zeros((0, 2)).astype(np.float32)
        all_level_anchors = []
        all_level_restrictions = []

        for idx, _ in enumerate(self.pyramid_levels):
            shifted_anchors = shift(featmap_sizes[idx],
                                    self.strides[idx],
                                    self.anchors[idx])
            restriction = generate_ranges(shifted_anchors.shape[0],
                                          self.gt_restrict_range[idx],
                                          self.gt_restrict_range[idx+1])

            # all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
            # all_restrictions = np.append(all_restrictions,
            #                              restriction, axis=0)
            all_level_anchors.append(torch.tensor(shifted_anchors,
                                                  dtype=dtype,
                                                  device=device))
            all_level_restrictions.append(torch.tensor(restriction,
                                                       dtype=dtype,
                                                       device=device))

        return all_level_anchors, all_level_restrictions

    def generate_anchors(self, base_size=16):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        """
        num_anchors = len(self.ratios) * len(self.scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(self.scales,
                                             (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios,
                                                  len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios,
                                                  len(self.scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors


def generate_ranges(anchor_number, min_length, max_length):
    restriction = np.zeros((anchor_number, 2)).astype(np.float32)
    restriction[:, 0] = min_length
    restriction[:, 1] = max_length

    return restriction


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


if __name__ == "__main__":
    anchors = Anchors()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.tensor([3.]).dtype
    temp = anchors([[30, 40], [18, 20], [3, 2], [1, 1], [1, 1]], dtype, device)
    pass

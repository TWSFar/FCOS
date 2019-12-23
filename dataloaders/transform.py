import cv2
import random
import numpy as np
import torch


class IrRegularResizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, min_side, max_side):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        H, W, _ = image.shape
        smallest_side = min(H, W)
        largest_side = max(H, W)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side
        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        new_size = (int(round(W*scale)), int(round((H*scale))))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        H, W, C = image.shape
        pad_w = 32 - W % 32 if W % 32 != 0 else 0
        pad_h = 32 - H % 32 if H % 32 != 0 else 0

        new_image = np.zeros((H + pad_h, W + pad_w, C)).astype(np.float32)
        new_image[:H, :W, :] = image.astype(np.float32)

        try:
            annots[:, :4] *= scale
        except:
            annots = np.array([])

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class RegularResizer(object):
    def __init__(self, input_size):
        self.input_size = input_size  # (x, y)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        ratio_x = self.input_size[0] / image.shape[1]
        ratio_y = self.input_size[1] / image.shape[0]
        image = cv2.resize(image,
                           (self.input_size[0], self.input_size[1]),
                           interpolation=cv2.INTER_LINEAR)
        annots[:, [0, 2]] = annots[:, [0, 2]] * ratio_x
        annots[:, [1, 3]] = annots[:, [1, 3]] * ratio_y

        W, H = self.input_size
        pad_w = 32 - W % 32 if W % 32 != 0 else 0
        pad_h = 32 - H % 32 if H % 32 != 0 else 0
        new_image = np.zeros((H + pad_h, W + pad_w, 3)).astype(np.float32)
        new_image[:H, :W, :] = image.astype(np.float32)

        image = torch.from_numpy(new_image)
        annots = torch.from_numpy(annots)

        return {'img': image, 'annot': annots, 'scale': (ratio_x, ratio_y)}


class Letterbox(object):
    """
    resize a rectangular image to a padded square
    """
    def __init__(self, input_size=(608, 608), train=True):
        self.input_size = input_size
        self.train = train

    def __call__(self, sample):
        assert self.input_size[0] == self.input_size[1], "input size is not square"
        image, annots = sample['img'], sample['annot']

        shape = image.shape[:2]  # shape = [height, width]
        ratio = float(self.input_size[0]) / max(shape)  # ratio  = old / new

        if self.train:
            dw = random.randint(0, max(shape) - shape[1])
            dh = random.randint(0, max(shape) - shape[0])
            left, right = dw, max(shape) - shape[1] - dw
            top, bottom = dh, max(shape) - shape[0] - dh
        else:
            dw = (max(shape) - shape[1]) / 2  # width padding
            dh = (max(shape) - shape[0]) / 2  # height padding
            left, right = round(dw - 0.1), round(dw + 0.1)
            top, bottom = round(dh - 0.1), round(dh + 0.1)

        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT)  # padded square
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))

        if annots is not None and len(annots) > 0:
            annots[:, 0] = ratio * (annots[:, 0] + left)
            annots[:, 1] = ratio * (annots[:, 1] + top)
            annots[:, 2] = ratio * (annots[:, 2] + left)
            annots[:, 3] = ratio * (annots[:, 3] + top)

        H, W, C = image.shape
        pad_w = 32 - W % 32 if W % 32 != 0 else 0
        pad_h = 32 - H % 32 if H % 32 != 0 else 0
        new_image = np.zeros((H + pad_h, W + pad_w, C)).astype(np.float32)
        new_image[:H, :W, :] = image.astype(np.float32)

        image = torch.from_numpy(new_image)
        annots = torch.from_numpy(annots)
        return {'img': image, 'annot': annots, 'scale': (ratio, left, top)}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            H, W, C = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = W - x2
            annots[:, 2] = W - x1

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']
        if image.max() > 1:
            image = image / 255.
        image = (image.astype(np.float32)-self.mean) / self.std

        return {'img': image, 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

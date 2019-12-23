import os
import os.path as osp
import cv2
import sys
import random
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders import transform as tsf

INSTANCES_SET = 'instances_{}.json'


class VisdroneDataset(Dataset):
    """Coco dataset."""

    def __init__(self, opt, set_name='train', train=True):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.opt = opt
        if train:
            self.root_dir = osp.join(opt.root_dir, 'VisDrone2019-DET-train')
        else:
            self.root_dir = osp.join(opt.root_dir, 'VisDrone2019-DET-val')
        self.anno_dir = osp.join(self.root_dir, 'annotations_json')
        self.img_dir = osp.join(self.root_dir, 'images')
        self.set_name = set_name
        self.train = train

        self.coco = COCO(osp.join(self.anno_dir, INSTANCES_SET.format(self.set_name)))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        self.min_size = opt.min_size
        self.max_size = opt.max_size
        self.input_size = (self.min_size, self.max_size)
        self.resize = self.resizes(opt.resize_type)
        self.train_tsf = transforms.Compose([
            tsf.Normalizer(),
            tsf.Augmenter(),
            self.resize
        ])

        self.test_tsf = transforms.Compose([
            tsf.Normalizer(),
            self.resize
        ])

    def resizes(self, resize_type):
        if resize_type == 'irregular':
            return tsf.IrRegularResizer(self.min_size, self.max_size)
        elif resize_type == 'regular':
            return tsf.RegularResizer(self.input_size)
        elif resize_type == "letterbox":
            return tsf.Letterbox(self.input_size, train=self.train)
        else:
            raise NotImplementedError

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        self.labels = {}
        for i, c in enumerate(categories):
            self.coco_labels[i] = c['id']
            self.coco_labels_inverse[c['id']] = i
            self.classes[c['name']] = i
            self.labels[i] = c['name']

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.train:
            sample = self.train_tsf(sample)
        else:
            sample = self.test_tsf(sample)
        sample['index'] = idx  # it is very import for val

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.img_dir, image_info['file_name'])
        # read img and BGR to RGB before normalize
        img = cv2.imread(path)[:, :, ::-1]
        return img.astype(np.float32)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    @property
    def num_classes(self):
        return 10

    @staticmethod
    def collater(data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]
        index = [s['index'] for s in data]

        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)

        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, "index": index}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass


if __name__ == '__main__':
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    opt = EasyDict()
    opt.root_dir = '/home/twsf/data/Visdrone'
    opt.batch_size = 2
    opt.input_size = (1024, 1024)
    opt.min_size = 1024
    opt.max_size = 1024
    dataset = VisdroneDataset(opt)
    print(dataset.labels)
    sample = dataset.__getitem__(0)
    sampler = AspectRatioBasedSampler(dataset, batch_size=2, drop_last=False)
    dl = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collater)
    for i, sp in enumerate(dl):
        pass

import os
import cv2
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.transform import UnNormalizer
# from dataloaders.utils import decode_seg_map_sequence

box_colors = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1),
              (0.541, 0.149, 0.341), (0.541, 0.169, 0.886),
              (0.753, 0.753, 0.753), (0.502, 0.165, 0.165),
              (0.031, 0.180, 0.329), (0.439, 0.502, 0.412),
              (0, 0, 0)  # others
              )


def plot_img(img, bboxes, id2name):
    for bbox in bboxes:
        try:
            if -1 in bbox:
                continue
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            id = int(bbox[4])
            label = id2name[id]

            if len(bbox) == 6:
                # if bbox[5] < 0.5:
                #     continue
                label = label + '|{:.2}'.format(bbox[5])

            # plot
            box_color = box_colors[min(id, len(box_colors)-1)]
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
            c1 = (x1, y1 - t_size[1] - 4)
            c2 = (x1 + t_size[0], y1)
            cv2.rectangle(img, c1, c2, color=box_color, thickness=-1)
            cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.4, (1, 1, 1), 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=2)

        except Exception as e:
            print(e)
            continue

    return img


class TensorboardSummary(object):
    def __init__(self, directory, opt):
        self.directory = directory
        self.unnor = UnNormalizer(opt.mean, opt.std)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, imgs, gts, outputs, id2name, global_step):
        # image transform((3, x, y) -> (x, y, 3) -> numpy -> BGR)
        gt_imgs = []
        pred_imgs = []
        show_len = min(len(imgs), 3)
        for idx in range(show_len):
            img = imgs[idx]
            img = self.unnor(img.cpu())
            img = img.permute(1, 2, 0).numpy()

            # gt bbox transform
            gt = gts[idx].cpu().numpy()
            gt_img = plot_img(img.copy(), gt, id2name)
            gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)
            gt_imgs.append(gt_img.unsqueeze(0))

            # predict bbox transform
            output = outputs[idx].cpu().numpy()
            pred_img = plot_img(img.copy(), output, id2name)
            pred_img = torch.from_numpy(pred_img).permute(2, 0, 1)
            pred_imgs.append(pred_img.unsqueeze(0))

        gt_imgs = torch.cat(gt_imgs, 0)
        pred_imgs = torch.cat(pred_imgs, 0)

        # target
        grid_target = make_grid(gt_imgs.clone().data, nrow=3, normalize=False)
        writer.add_image('Groundtruth density', grid_target, global_step)

        # output
        grid_output = make_grid(pred_imgs.clone().data, nrow=3, normalize=False)
        writer.add_image('Predicted density', grid_output, global_step)

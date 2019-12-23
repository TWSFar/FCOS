import os
import fire
import json
import collections
import numpy as np

from dataloaders import make_data_loader
from models.retinanet import RetinaNet
from models.utils.functions import PostProcess
from models_demo import model_demo
from utils.visdrone_config import opt
from utils.visualization import TensorboardSummary
from utils.saver import Saver

import torch
import torch.optim as optim

from pycocotools.cocoeval import COCOeval

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self):
        # Define Saver
        self.saver = Saver(opt)
        self.saver.save_experiment_config()

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Define Dataloader
        # train dataset
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)
        self.num_bt_tr = len(self.train_loader)
        self.num_classes = self.train_dataset.num_classes

        # val dataset
        self.val_dataset, self.val_loader = make_data_loader(opt, train=False)
        self.num_bt_val = len(self.val_loader)

        # Define Network
        # initilize the network here.
        # self.model = RetinaNet(opt, self.num_classes)
        self.model = model_demo.resnet50(num_classes=10, pretrained=False)
        self.model = self.model.to(opt.device)

        # contain nms for val
        self.post_pro = PostProcess(opt.pst_thd, opt.nms_thd)

        # Define Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)

        # Define lr scherduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, verbose=True)

        # Resuming Checkpoint
        self.best_pred = 0.0
        self.start_epoch = opt.start_epoch
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                opt.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        # Using mul gpu
        if len(opt.gpu_id) > 1:
            print("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        self.loss_hist = collections.deque(maxlen=500)

    def training(self, epoch):
        self.model.train()
        if len(opt.gpu_id) > 0:
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(self.train_loader):
            try:
                self.optimizer.zero_grad()
                imgs = data['img'].to(opt.device)
                target = data['annot'].to(opt.device)

                cls_loss, loc_loss = self.model([imgs, target])

                cls_loss = cls_loss.mean()
                loc_loss = loc_loss.mean()
                loss = cls_loss + loc_loss

                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # visualize
                global_step = iter_num + self.num_bt_tr * epoch
                self.writer.add_scalar('train/cls_loss_epoch', cls_loss.cpu().item(), global_step)
                self.writer.add_scalar('train/loc_loss_epoch', loc_loss.cpu().item(), global_step)

                if global_step % opt.print_freq == 0:
                    printline = 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'
                    print(printline.format(
                        epoch, iter_num,
                        float(cls_loss), float(loc_loss),
                        np.mean(self.loss_hist)))

                del cls_loss
                del loc_loss

            except Exception as e:
                print(e)
                continue

        self.scheduler.step(np.mean(epoch_loss))

    def validate(self, epoch):
        self.model.eval()
        # start collecting results
        with torch.no_grad():
            results = []
            image_ids = []
            for ii, data in enumerate(self.val_loader):
                scale = data['scale'][0]
                index = data['index'][0]
                img = data['img'].to(opt.device).float()
                target = data['annot']

                # run network
                scores, labels, boxes = self.model(img)

                scores, labels, boxes = self.post_pro(scores, labels, boxes)
                continue
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()

                # visualize
                global_step = ii + self.num_bt_val * epoch
                if global_step % opt.plot_every == 0:
                    ouput = torch.cat((boxes, labels.float().unsqueeze(1), scores.unsqueeze(1)), dim=1)
                    self.summary.visualize_image(
                        self.writer,
                        img[0], target[0], ouput,
                        self.val_dataset.labels,
                        global_step)

                # correct boxes for image scale
                boxes = boxes / scale

                if boxes.shape[0] > 0:
                    # change to (x, y, w, h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # compute predicted labels and scores
                    # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < opt.pst_thd:
                            break

                        # append detection for each positively labeled class
                        image_result = {
                            'image_id': self.val_dataset.image_ids[index],
                            'category_id': self.val_dataset.label_to_coco_label(label),
                            'score': float(score),
                            'bbox': box.tolist(),
                        }

                        # append detection to results
                        results.append(image_result)

                # append image to list of processed images
                image_ids.append(self.val_dataset.image_ids[index])

                # print progress
                print('{}/{}'.format(ii, len(self.val_dataset)), end='\r')

            if not len(results):
                return

            # write output
            json.dump(results, open('run/coco/{}_bbox_results.json'.format(self.val_dataset.set_name), 'w'), indent=4)

            # load results in COCO evaluation tool
            coco_true = self.val_dataset.coco
            coco_pred = coco_true.loadRes('run/coco/{}_bbox_results.json'.format(self.val_dataset.set_name))

            # run COCO evaluation
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            return


def eval(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()
    print('Num evaluating images: {}'.format(len(trainer.val_dataset)))

    trainer.validate(trainer.start_epoch)


def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()

    print('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(opt.start_epoch, opt.epochs):
        # train
        trainer.training(epoch)

        # val
        trainer.validate(epoch)

        if (epoch % opt.saver_freq == 0):
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best=False)


if __name__ == '__main__':
    # train()
    fire.Fire()

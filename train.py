import os
import fire
import json
import time
import collections
import numpy as np

# from models_demo import model_demo

from configs.fcos_res50_visdrone import opt
# from configs.visdrone_chip import opt
# from configs.visdrone_samples import opt
# from configs.coco import opt
# from configs.retina_visdrone import opt

from dataloaders import make_data_loader
from models import Model
from models.utils import (PostProcess, DefaultEval,
                          re_resize, parse_losses)
from utils import TensorboardSummary, Saver, Timer

import torch
import torch.optim as optim
from pycocotools.cocoeval import COCOeval

import multiprocessing
multiprocessing.set_start_method('spawn', True)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)


class Trainer(object):
    def __init__(self, mode):
        # Define Saver
        self.saver = Saver(opt, mode)

        # visualize
        self.summary = TensorboardSummary(self.saver.experiment_dir, opt)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        # train dataset
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)
        self.nbatch_train = len(self.train_loader)
        self.num_classes = self.train_dataset.num_classes

        # val dataset
        self.val_dataset, self.val_loader = make_data_loader(opt, train=False)
        self.nbatch_val = len(self.val_loader)

        # Define Network
        # initilize the network here.
        self.model = Model(opt, self.num_classes)
        # self.model = RetinaNet(opt, self.num_classes)
        self.model = self.model.to(opt.device)

        # contain nms for val
        self.post_pro = PostProcess(**opt.nms)

        # Define Optimizer
        if opt.adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.decay)

        # Define lr scherduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, verbose=True)

        # Resuming Checkpoint
        self.best_pred = 0.0
        self.start_epoch = 0

        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.start_epoch = checkpoint['epoch'] + 1
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
        self.timer = Timer(opt.epochs, self.nbatch_train, self.nbatch_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def training(self, epoch):
        self.model.train()
        if len(opt.gpu_id) > 1:
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(self.train_loader):
            if iter_num > 3: break
            try:
                temp_time = time.time()
                self.optimizer.zero_grad()
                imgs = data['img'].to(opt.device)
                targets = data['annot'].to(opt.device)

                losses = self.model(imgs, targets)
                loss, log_vars = parse_losses(losses)

                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                self.optimizer.step()
                self.loss_hist.append(float(loss.cpu().item()))
                epoch_loss.append(float(loss.cpu().item()))

                # visualize
                global_step = iter_num + self.nbatch_train * epoch + 1
                loss_logs = ""
                for _key, _value in log_vars.items():
                    loss_logs += "{}: {:.4f}  ".format(_key, _value)
                    self.writer.add_scalar('train/{}'.format(_key),
                                           _value,
                                           global_step)

                batch_time = time.time() - temp_time
                eta = self.timer.eta(global_step, batch_time)
                self.step_time.append(batch_time)
                if global_step % opt.print_freq == 0:
                    printline = ("Epoch: [{}][{}/{}]  "
                                 "lr: {}  eta: {}  time: {:1.3f}  "
                                 "{}"
                                 "Running loss: {:1.5f}").format(
                                    epoch, iter_num + 1, self.nbatch_train,
                                    self.optimizer.param_groups[0]['lr'],
                                    eta, np.sum(self.step_time),
                                    loss_logs,
                                    np.mean(self.loss_hist))
                    print(printline)
                    self.saver.save_experiment_log(printline)

            except Exception as e:
                print(e)
                continue

        self.scheduler.step(np.mean(epoch_loss))

    def validate(self, epoch):
        self.model.eval()
        def_eval = DefaultEval()
        # start collecting results
        with torch.no_grad():
            results = []
            image_ids = []
            for ii, data in enumerate(self.val_loader):
                # if ii > 2: break
                scale = data['scale']
                index = data['index']
                imgs = data['img'].to(opt.device).float()
                targets = data['annot']

                # run network
                scores, labels, boxes = self.model(imgs)

                scores_bt, labels_bt, boxes_bt = self.post_pro(
                    scores, labels, boxes, imgs.shape[-2:])

                outputs = []
                for k in range(len(boxes_bt)):
                    outputs.append(torch.cat((
                        boxes_bt[k],
                        labels_bt[k].unsqueeze(1).float(),
                        scores_bt[k].unsqueeze(1)),
                        dim=1))

                # statistics
                if opt.eval_type == "default":
                    def_eval.statistics(outputs, targets, iou_thresh=0.5)

                # visualize
                global_step = ii + self.nbatch_val * epoch
                if global_step % opt.plot_every == 0:
                    self.summary.visualize_image(
                        self.writer,
                        imgs, targets, outputs,
                        self.val_dataset.labels,
                        global_step)

                if opt.eval_type == "cocoeval":
                    # save json
                    for jj in range(len(boxes_bt)):
                        pre_bboxes = boxes_bt[jj]
                        pre_scrs = scores_bt[jj]
                        pre_labs = labels_bt[jj]

                        if pre_bboxes.shape[0] > 0:
                            re_resize(pre_bboxes, scale, opt.resize_type)

                            # change to (x, y, w, h) (MS COCO standard)
                            pre_bboxes[:, 2] -= pre_bboxes[:, 0]
                            pre_bboxes[:, 3] -= pre_bboxes[:, 1]

                            # compute predicted labels and scores
                            for box_id in range(pre_bboxes.shape[0]):
                                score = float(pre_scrs[box_id])
                                label = int(pre_labs[box_id])
                                box = pre_bboxes[box_id, :]
                                # append detection for each positively labeled class
                                image_result = {
                                    'image_id': self.val_dataset.image_ids[index[jj]],
                                    'category_id': self.val_dataset.label_to_coco_label(label),
                                    'score': float(score),
                                    'bbox': box.tolist(),
                                }

                                # append detection to results
                                results.append(image_result)

                # append image to list of processed images
                for idx in index:
                    image_ids.append(self.val_dataset.image_ids[idx])

                # print progress
                print('{}/{}'.format(ii, len(self.val_loader)), end='\r')

            if opt.eval_type == "default":
                # Compute statistics
                stats = [np.concatenate(x, 0) for x in list(zip(*def_eval.stats))]
                # number of targets per class
                nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_classes)
                if len(stats):
                    p, r, ap, f1, ap_class = def_eval.ap_per_class(*stats)
                    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

                # visualize
                titles = ['Precision', 'Recall', 'mAP', 'F1']
                result = [mp, mr, map, mf1]
                for xi, title in zip(result, titles):
                    self.writer.add_scalar('val/{}'.format(title), xi, epoch)

                # Print and Write results
                title = ('%10s' * 7) % ('epoch: [{}]'.format(epoch), 'Class', 'Targets', 'P', 'R', 'mAP', 'F1')
                print(title)
                self.saver.save_eval_result(stats=title)
                printline = '%20s' + '%10.3g' * 5
                pf = printline % ('all', nt.sum(), mp, mr, map, mf1)  # print format
                print(pf)
                self.saver.save_eval_result(stats=pf)
                if self.num_classes > 1 and len(stats):
                    for i, c in enumerate(ap_class):
                        pf = printline % (self.val_dataset.labels[c], nt[c], p[i], r[i], ap[i], f1[i])
                        print(pf)
                        self.saver.save_eval_result(stats=pf)

                return map

            elif opt.eval_type == "cocoeval":
                # write output
                if not len(results):
                    return 0
                json.dump(results, open('run/{}/{}_bbox_results.json'.format(
                    opt.dataset, self.val_dataset.set_name), 'w'), indent=4)

                # load results in COCO evaluation tool
                coco_true = self.val_dataset.coco
                coco_pred = coco_true.loadRes('run/{}/{}_bbox_results.json'.format(
                    opt.dataset, self.val_dataset.set_name))

                # run COCO evaluation
                coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
                coco_eval.params.imgIds = image_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # save result
                stats = coco_eval.stats
                self.saver.save_coco_eval_result(stats=stats, epoch=epoch)

                # visualize
                self.writer.add_scalar('val/AP50', stats[1], epoch)

                # according AP50
                return stats[1]

            else:
                raise NotImplementedError


def eval(**kwargs):
    opt._parse(kwargs)
    evaler = Trainer("val")
    print('Num evaluating images: {}'.format(len(evaler.val_dataset)))

    evaler.validate(evaler.start_epoch)


def train(**kwargs):
    start_time = time.time()
    opt._parse(kwargs)
    trainer = Trainer("train")

    print('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(trainer.start_epoch, opt.epochs):
        # train
        trainer.training(epoch)

        # val
        val_time = time.time()
        ap50 = trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        is_best = ap50 > trainer.best_pred
        trainer.best_pred = max(ap50, trainer.best_pred)
        if (epoch % opt.saver_freq == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    all_time = trainer.timer.second2hour(time.time()-start_time)
    print("Train done!, Sum time: {}, Best result: {}".format(all_time, trainer.best_pred))

    # cache result
    print("Backup result...")
    trainer.saver.backup_result()
    print("Done!")


if __name__ == '__main__':
    # train()
    fire.Fire(train)

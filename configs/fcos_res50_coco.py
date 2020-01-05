import os
import time
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "coco"
    root_dir = user_dir + "/work/RetinaNet/data/COCO"
    resume = False
    resize_type = "letterbox"  # [regular, irregular, letterbox]
    min_size = 1024
    max_size = 1024
    pre = '/home/twsf/work/RetinaNet/run/visdrone_chip/20191123_115838/model_best.pth.tar'

    # model
    backbone = 'resnet50'
    strides = [8, 16, 32, 64, 128]
    # head
    head = dict(
        type="FCOSHead",
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0))

    # train
    batch_size = 2
    epochs = 40
    workers = 1

    # optimizer
    adam = True
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    gamma = 0.3

    # eval
    eval_type = "default"
    nms = dict(
        type="GreedyNms",  # SoftNms
        pst_thd=0.2,
        nms_thd=0.5,
        n_pre_nms=20000
    )

    # visual
    visualize = True
    print_freq = 50
    plot_every = 200  # every n batch plot
    saver_freq = 1

    seed = int(time.time())

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.device, self.gpu_id = select_device()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
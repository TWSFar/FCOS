import os
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = user_dir + "/data/Visdrone"
    resize_type = "letterbox"  # [regular, irregular, letterbox]
    min_size = 1024
    max_size = 1024
    mean = [0.373, 0.378, 0.365]
    std = [0.192, 0.183, 0.194]
    resume = False
    pre = None

    # model
    model = "retina"
    backbone = 'resnet50'
    neck = "fpn"
    # head
    head = dict(
        type="RetinaHead",
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='CIoULoss')
    )

    # train
    batch_size = 3
    epochs = 70
    workers = 1
    freeze_bn = True

    # optimizer
    adam = True
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    scales = 0.3

    # eval
    eval_type = "default"  # [cocoeval, default]
    nms = dict(
        type="GreedyNms",  # SoftNms
        pst_thd=0.2,
        nms_thd=0.5,
        n_pre_nms=20000
    )

    # visual
    print_freq = 10
    plot_every = 50  # every n batch plot
    saver_freq = 1

    seed = 1

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

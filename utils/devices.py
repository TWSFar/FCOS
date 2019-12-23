import torch
import numpy as np


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    gpu_num = torch.cuda.device_count()

    if not cuda:
        print('Using CPU\n')
    if cuda:
        c = 1024 ** 2
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_num)]
        for i in range(gpu_num):
            print('Using CUDA device{} _CudaDeviceProperties(name={}, total_memory={}MB'.\
                  format(i, x[i].name, round(x[i].total_memory/c)))
        print('')

    return device, np.arange(0, gpu_num).tolist()
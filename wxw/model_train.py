import os
import math
import torch
import random
import numpy as np
from torch import nn
from collections import defaultdict
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR


class LogHistory:
    def __init__(self, log_dir='./'):
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses = defaultdict(list)
        self.writer = SummaryWriter(self.log_dir)
        self.plt_colors = ["red", "yellow", "black", "blue"]

    def add_graph(self, model, size):
        try:
            dummy_input = torch.randn((2, 3, size[0], size[1]))
            self.writer.add_graph(model, dummy_input)
        except Exception as why:
            print("dummy model: ")
            print(why)
            pass

    def update_info(self, info, step):
        for name, value in info.items():
            if np.isscalar(value):
                self.writer.add_scalar(name, value, step)
            elif torch.is_tensor(value):
                self.writer.add_images(name, value, step)
            elif type(value) in [list, tuple]:
                self.add_str_list(name, value, step)
            else:
                raise TypeError(f"{name}: {type(value)}, {value}")

    def add_str_list(self, name, values, step):
        for idx, value in enumerate(values):
            if not isinstance(value, str):
                continue
            self.writer.add_text(f"{name}_{idx}", value, step)


def set_random_seed(seed=123456, deterministic=False):
    """ Set random state to random libray, numpy, torch and cudnn.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(
        v for k, v in torch.nn.__dict__.items() if "Norm" in k
    )  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)
    if name == "Adam":
        optimizer = torch.optim.Adam(
            g[2], lr=lr, betas=(momentum, 0.999)
        )  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(
            g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group(
        {"params": g[0], "weight_decay": decay}
    )  # add g0 with weight_decay
    optimizer.add_param_group(
        {"params": g[1], "weight_decay": 0.0}
    )  # add g1 (BatchNorm2d weights)
    return optimizer


def smart_lr(warm, lr_init, lrf, epochs, cycle):
    epochs -= 1  # last_step begin with 0
    if cycle:
        y1, y2 = 1.0, lrf
        return (
            lambda t: t / warm
            if t < warm
            else ((1 - np.cos((t - warm) / (epochs - warm) * np.pi)) / 2) * (y2 - y1)
                 + y1
        )
    else:
        lr_min = lr_init * lrf
        return (
            lambda x: x / warm
            if 0 < x < warm
            else (1 - (x - warm) / (epochs - warm)) * (1 - lr_min) + lr_min
        )


def warmup_cos_lr(optimizer, warmup_epochs, total_epochs):
    # 定义学习率调度器的lambda函数，用于实现warmup
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            return 0.5 * (
                    1 + math.cos((current_epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-4) * math.pi))

    return LambdaLR(optimizer, lr_lambda)


def weights_init(m):
    normal_layers = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    bn_layers = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    if isinstance(m, normal_layers):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, bn_layers):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def compute_gflops(model, input_size):
    # =========FLOPS===========
    from thop import profile
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    inputs = torch.randn(1, 3, *input_size).to(model.device)
    macs, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


def compute_onnx_gflops(path):
    import onnx_tool
    onnx_tool.model_profile(path)

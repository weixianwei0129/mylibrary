import torch
import numpy as np
import torch.backends.cudnn as cudnn


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


class LogHistory:
    def __init__(self, log_dir):
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

    def update_info(self, info, epoch):
        for name, value in info.items():
            if np.isscalar(value):
                self.add_scalar(name, value, epoch)
            elif type(value) in [torch.Tensor]:
                self.add_image(name, value, epoch)
            else:
                raise TypeError(f"{name}: {type(value)}, {value}")

    def add_scalar(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)

    def add_image(self, name, images, epoch):
        self.writer.add_images(name, images, epoch)


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

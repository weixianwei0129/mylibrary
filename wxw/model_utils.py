import gc
import os
import math
import random
from collections import defaultdict

import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class LogHistory:
    """Class to log training history using TensorBoard.

    Attributes:
        log_dir (str): Directory to save logs.
        losses (defaultdict): Dictionary to store loss values.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        plt_colors (list): List of colors for plotting.
    """

    def __init__(self, log_dir='./'):
        """Initialize LogHistory with a specified log directory.

        Args:
            log_dir (str): Directory to save logs. Defaults to './'.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses = defaultdict(list)
        self.writer = SummaryWriter(self.log_dir)
        self.plt_colors = ["red", "yellow", "black", "blue"]

    def add_graph(self, model, size):
        """Add a model graph to TensorBoard.

        Args:
            model (torch.nn.Module): The model to be added.
            size (tuple): Size of the input tensor.
        """
        try:
            dummy_input = torch.randn((2, 3, size[0], size[1]))
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print("Error adding model graph: ", e)

    def update_info(self, info, step):
        """Update TensorBoard with new information.

        Args:
            info (dict): Dictionary containing information to log.
            step (int): Current step or epoch.
        """
        for name, value in info.items():
            if np.isscalar(value):
                self.writer.add_scalar(name, value, step)
            elif torch.is_tensor(value):
                self.writer.add_images(name, value, step)
            elif isinstance(value, (list, tuple)):
                self.add_str_list(name, value, step)
            else:
                raise TypeError(f"Unsupported type for {name}: {type(value)}, {value}")

    def add_str_list(self, name, values, step):
        """Add a list of strings to TensorBoard.

        Args:
            name (str): Name of the text list.
            values (list): List of string values.
            step (int): Current step or epoch.
        """
        for idx, value in enumerate(values):
            if isinstance(value, str):
                self.writer.add_text(f"{name}_{idx}", value, step)


def set_random_seed(seed=123456, deterministic=False):
    """Set random state for random, numpy, torch, and cudnn libraries.

    Args:
        seed (int): Seed value for random number generators.
        deterministic (bool): Whether to set deterministic mode for cudnn.
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
    """Create an optimizer with specific parameter groups for a model.

    Args:
        model (torch.nn.Module): The model to optimize.
        name (str): The name of the optimizer to use.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        decay (float): Weight decay (L2 penalty).

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    param_groups = ([], [], [])  # optimizer parameter groups
    norm_layers = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=0):
            if param_name == "bias":  # bias (no decay)
                param_groups[2].append(param)
            elif param_name == "weight" and isinstance(module, norm_layers):  # weight (no decay)
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(param_groups[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(param_groups[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(param_groups[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(param_groups[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": param_groups[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": param_groups[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer


def smart_lr(warmup_epochs, initial_lr, final_lr_factor, total_epochs, use_cycle):
    """Generate a learning rate scheduler function.

    This function returns a lambda function that calculates the learning rate
    based on the current epoch, with options for warmup and cyclical learning rates.

    Args:
        warmup_epochs (int): Number of warmup epochs.
        initial_lr (float): Initial learning rate.
        final_lr_factor (float): Factor to multiply the initial learning rate for the final learning rate.
        total_epochs (int): Total number of epochs.
        use_cycle (bool): Whether to use a cyclical learning rate.

    Returns:
        function: A lambda function that calculates the learning rate for a given epoch.
    """
    total_epochs -= 1  # Adjust for zero-based indexing

    if use_cycle:
        y1, y2 = 1.0, final_lr_factor
        return (
            lambda epoch: epoch / warmup_epochs
            if epoch < warmup_epochs
            else ((1 - np.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * np.pi)) / 2) * (y2 - y1) + y1
        )
    else:
        min_lr = initial_lr * final_lr_factor
        return (
            lambda epoch: epoch / warmup_epochs
            if 0 < epoch < warmup_epochs
            else (1 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)) * (1 - min_lr) + min_lr
        )


def warmup_cos_lr(optimizer, warmup_epochs, total_epochs):
    """Create a learning rate scheduler with warmup and cosine annealing.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_epochs (int): Number of epochs for the warmup phase.
        total_epochs (int): Total number of epochs for training.

    Returns:
        LambdaLR: A learning rate scheduler.
    """

    def lr_lambda(current_epoch):
        """Lambda function to calculate the learning rate multiplier."""
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            return 0.5 * (
                    1 + math.cos((current_epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-4) * math.pi)
            )

    return LambdaLR(optimizer, lr_lambda)


def weights_init(m):
    """Initialize weights for layers in a neural network.

    This function initializes weights for convolutional, transposed convolutional,
    linear, and batch normalization layers using specific initialization methods.

    Args:
        m (nn.Module): A layer in a neural network.
    """
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
    if 'Conv' in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def torch_model_gflops(model, x):
    """Calculate and print the GFLOPs of a PyTorch model using fvcore.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
        x (torch.Tensor): The input tensor for the model.
    """
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print("FLOPs: ", flops.total() / 1e9)
    print(flop_count_table(flops))


# ========onnx======

class ONNXRunner:
    """
    A class to run ONNX models using ONNX Runtime.

    Attributes:
        session (onnxruntime.InferenceSession): The ONNX Runtime session for inference.
    """

    def __init__(self, path):
        """
        Initializes the ONNXRunner with the given model path.

        Args:
            path (str): The path to the ONNX model file.
        """
        import onnxruntime
        providers = [
            "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"
        ]
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        print("Inputs: ", [input.name for input in self.session.get_inputs()])
        print("Outputs: ", [output.name for output in self.session.get_outputs()])

    def __call__(self, img):
        """
        Runs inference on the provided image.

        Args:
            img (numpy.ndarray): The input image for inference.

        Returns:
            list: The inference results.
        """
        try:
            return self.session.run(
                [output.name for output in self.session.get_outputs()],
                {self.session.get_inputs()[0].name: img},
            )
        except Exception as e:
            print("[ONNXRunner] Error during inference:")
            print(e)
            print("Input details:", self.session.get_inputs()[0])
            print("Image shape:", img.shape)


def onnx_model_gflops(path):
    """Calculate and print the GFLOPs of an ONNX model.

    Args:
        path (str): The path to the ONNX model file.
    """
    import onnx_tool
    onnx_tool.model_profile(path)

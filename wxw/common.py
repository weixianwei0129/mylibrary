import argparse
import base64
import collections
import functools
import glob
import hashlib
import json
import os
import os.path as osp
import random
import shutil
import time
import warnings
from collections import defaultdict
from functools import partial
from io import BytesIO
from multiprocessing import Pool
import yaml
import cv2
import matplotlib.font_manager as fm
import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pl_version
from functools import wraps

# Set random seeds for reproducibility
np.random.seed(123456)
random.seed(123456)


# ========inference=====

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


# =============基础方法===============
# Terminal color output functions
def print_red(text):
    """Print text in red color."""
    print(f"\033[31m{text}\033[0m")


def print_green(text):
    """Print text in green color."""
    print(f"\033[32m{text}\033[0m")


def print_yellow(text):
    """Print text in yellow color."""
    print(f"\033[33m{text}\033[0m")


def print_blue(text):
    """Print text in blue color."""
    print(f"\033[34m{text}\033[0m")


def update_args(old_, new_) -> argparse.Namespace:
    """
    Update the arguments from old_ with new_.

    Args:
        old_ (Union[argparse.Namespace, str, dict]): The original arguments.
        new_ (Union[argparse.Namespace, str, dict]): The new arguments to update with.

    Returns:
        argparse.Namespace: The updated arguments as a Namespace object.
    """
    if isinstance(old_, argparse.Namespace):
        old_ = vars(old_)
    elif isinstance(new_, argparse.Namespace):
        new_ = vars(new_)

    if isinstance(old_, str) and old_.endswith('.yaml'):
        with open(old_, 'r') as file:
            old_ = yaml.safe_load(file)
    elif isinstance(new_, str) and new_.endswith('.yaml'):
        with open(new_, 'r') as file:
            new_ = yaml.safe_load(file)

    assert isinstance(old_, dict) and isinstance(new_, dict)
    old_.update(new_)
    return argparse.Namespace(**old_)


def safe_replace(src: str, _old, _new) -> str:
    """
    Safely replace occurrences of _old with _new in src.

    Args:
        src (str): The source string.
        _old (list or str): The substring to be replaced.
        _new (list or str): The substring to replace with.

    Returns:
        str: The modified string, or None if no replacement was made.
    """
    if isinstance(_old, str):
        _old = [_old]
        _new = [_new]
    assert len(_old) == len(_new)
    for _o, _n in zip(_old, _new):
        dst = src.replace(_o, _n)
    if dst == src:
        print("No replacement made!")
        return None
    return dst


def md5sum(file_path: str) -> str:
    """
    Calculate the MD5 checksum of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The MD5 checksum of the file.
    """
    with open(file_path, "rb") as file:
        md5_hash = hashlib.md5()
        while True:
            data = file.read(4096)  # 每次读取4KB数据
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()


def print_format(string: str, a: float, func: str, b: float) -> float:
    """
    Format and print a mathematical operation, then return the result.

    Args:
        string (str): The description of the operation.
        a (float): The first operand.
        func (str): The operator as a string (e.g., '+', '-', '*', '/').
        b (float): The second operand.

    Returns:
        float: The result of the operation.
    """
    formatted_string = f"{a:<5.3f} {func} {b:<5.3f}"
    if func == '/':
        b += 1e-4  # Avoid division by zero
    c = eval(f"{a} {func} {b}")
    print(f"{string:<20}: {formatted_string} = {c:.3f}")
    return c


def divisibility(a: float, r: int = 32) -> int:
    """
    计算a是否能被r整除，如果不能则返回大于a的最小r的倍数。

    Args:
        a (float): 要检查的数字。
        r (int, optional): 用于整除的基数，默认为32。

    Returns:
        int: 大于或等于a的最小r的倍数。
    """
    if r == 1:
        return int(a)
    return int(np.ceil(a / r) * r)


def get_offset_coordinates(start_point, end_point, min_value: float, max_value: float):
    """
    Adjust the start and end points of a line segment to ensure they fall within the specified range.
    If the length of the line segment is greater than the range, a warning is printed and the original points are returned.

    Args:
        start_point (float): The initial start point of the line segment.
        end_point (float): The initial end point of the line segment.
        min_value (float): The minimum allowable value.
        max_value (float): The maximum allowable value.

    Returns:
        tuple: The adjusted start and end points of the line segment.
    """
    if end_point - start_point > max_value - min_value:
        print(
            f"[get_offset_coordinates] warning: "
            f"end_point - start_point > max_value - min_value: "
            f"{end_point - start_point} > {max_value - min_value}"
        )
        return start_point, end_point

    end_offset = max([0, min_value - start_point])
    start_point = max(min_value, start_point)
    start_offset = max([0, end_point - max_value])
    end_point = min(max_value, end_point)
    start_point = max(start_point - start_offset, min_value)
    end_point = min(end_point + end_offset, max_value)

    return start_point, end_point


def cost_time(func):
    """
    Decorator that measures the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with execution time measurement.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = (time.perf_counter() - start_time) * 1000
        print(f"[INFO] [{func.__name__}] cost time: {elapsed_time:.4f}ms")
        return result

    return wrapper


def xywh2xyxy(pts):
    """
    Convert bounding boxes from (center x, center y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        pts (np.ndarray or list): Array of bounding boxes in (cx, cy, w, h) format.

    Returns:
        np.ndarray: Array of bounding boxes in (x1, y1, x2, y2) format.
    """
    pts = np.reshape(pts, [-1, 4])
    cx, cy, w, h = np.split(pts, 4, 1)
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    res = np.concatenate([x1, y1, x2, y2], axis=1)
    res = np.clip(res, 0, np.inf)
    return np.squeeze(res)


def xyxy2xywh(pts):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (center x, center y, width, height) format.

    Args:
        pts (np.ndarray or list): Array of bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        np.ndarray: Array of bounding boxes in (cx, cy, w, h) format.
    """
    pts = np.reshape(pts, [-1, 4])
    x1, y1, x2, y2 = np.split(pts, 4, 1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = np.abs(x1 - x2)
    h = np.abs(y1 - y2)
    res = np.concatenate([cx, cy, w, h], axis=1)
    res = np.clip(res, 0, np.inf)
    return np.squeeze(res)


def get_min_rect(pts):
    """
    Get the minimum bounding rectangle for a set of points.

    Args:
        pts (np.ndarray or list): Array of points with shape (N, 2).

    Returns:
        np.ndarray: Array containing [x_min, y_min, x_max, y_max, cx, cy, w, h].
    """
    pts = np.reshape(pts, (-1, 2))
    x_min = min(pts[:, 0])
    x_max = max(pts[:, 0])
    y_min = min(pts[:, 1])
    y_max = max(pts[:, 1])
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return np.array([x_min, y_min, x_max, y_max, cx, cy, w, h])


def clockwise_points(pts):
    """
    Sort points in clockwise order.

    Args:
        pts (list): List of points in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].

    Returns:
        list: List of points sorted in clockwise order.

    1. 先按照x进行排序,从小到大
    2. 对前两个点,y大的为点4, 小的为点1
    3. 对后两个点,y大的为点3, 小的为点2
    """
    pts = sorted(pts, key=lambda x: x[0])

    if pts[1][1] > pts[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if pts[3][1] > pts[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2
    return [pts[index_1], pts[index_2], pts[index_3], pts[index_4]]


def generate_random_color(min_value, max_value):
    """Generate a random color.

    Args:
        min_value (int): The minimum value for the color components.
        max_value (int): The maximum value for the color components.

    Returns:
        tuple: A tuple containing three integers representing the blue, green, and red components of the color.
    """
    blue = np.random.randint(min_value, max_value)
    green = np.random.randint(min_value, max_value)
    red = np.random.randint(min_value, max_value)
    return blue, green, red


def create_color_list(num_colors):
    """Create a list of colors.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        np.ndarray: An array of RGB color values.
    """
    if num_colors < 10:
        colors = np.array(plt.cm.tab10.colors)
    else:
        colors = np.array(plt.cm.tab20.colors)

    colors = (colors[:num_colors - 1, ::-1] * 255)
    colors = np.insert(colors, 0, (0, 0, 0), axis=0)
    return [tuple(x) for x in colors]


def multi_process(process_method, data_to_process, num_threads=1):
    """Run a method in multiple processes.

    Args:
        process_method (function): The method to be run in multiple processes.
        data_to_process (list): The data to be processed.
        num_threads (int, optional): The number of threads to use. Defaults to 1.

    Example:
        def process_method(args):
            thread_idx, data_to_process = args
            # Processing code here

        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
        multi_process(process_method, data_to_process, num_threads=4)
    """
    if num_threads == 1:
        process_method([0, data_to_process])
    else:
        total = len(data_to_process)
        interval = int(np.ceil(total / num_threads))

        tasks = []
        for i in range(num_threads):
            start = i * interval
            end = min(start + interval, total)
            tasks.append([i, data_to_process[start:end]])

        with Pool(num_threads) as pool:
            pool.map(process_method, tasks)


def simplify_number(decimal):
    """Convert a decimal number to its simplest fraction form.

    Args:
        decimal (float): The decimal number to be converted.

    Returns:
        tuple: A tuple containing the fraction as a string, the numerator, and the denominator.
    """
    from fractions import Fraction

    # Convert the decimal to the simplest fraction
    fraction = Fraction(decimal).limit_denominator()

    # Create a string representation of the fraction
    fraction_string = f"{fraction.numerator}/{fraction.denominator}"

    return fraction_string, fraction.numerator, fraction.denominator


# =========Files: 文件移动和写入============
def merge_path(path, flag, ignore=None):
    """Merge a path from a specific flag.

    Args:
        path (str): The original path.
        flag (str): The flag to start merging from.
        ignore (list): Ignore parts

    Returns:
        str: The merged path starting from the flag.
    """
    if ignore is None:
        ignore = []
    path_parts = path.split(os.sep)
    flag_index = path_parts.index(flag)
    path_parts = path_parts[flag_index:]
    for i, x in enumerate(ignore):
        path_parts.pop(x + i)
    return '_'.join(path_parts)


def move_file_pair(
        path,
        dst_folder,
        dst_name=None,
        postfixes=None,
        copy=True,
        execute=False,
        empty_undo=False,
        empty_delete=False,
        ignore_failed=False,
        overwrite=False,
):
    """Move or copy file pairs to a destination folder.

    Args:
        path (str): The source file path.
        dst_folder (str): The destination folder.
        dst_name (str, optional): The destination file name without postfix. Defaults to None.
        postfixes (list, optional): List of postfixes to consider. Defaults to None.
        copy (bool, optional): Whether to copy instead of move. Defaults to True.
        execute (bool, optional): Whether to execute the move/copy. Defaults to False.
        empty_undo (bool, optional): Whether to undo if the file is empty. Defaults to False.
        empty_delete (bool, optional): Whether to delete the file if it is empty. Defaults to False.
        ignore_failed (bool, optional): Whether to ignore the options that failed to move or copy. Defaults to False.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to True.
        

    Returns:
        None
    """
    if empty_undo and os.path.getsize(path) == 0:
        if empty_delete:
            os.remove(path)
        return

    prefix, self_postfix = osp.splitext(path)
    if postfixes is None:
        postfixes = [self_postfix]

    src_dir = osp.dirname(prefix)
    src_name = osp.basename(prefix)

    if dst_name is None:
        dst_name = src_name
    else:
        # simple check
        for postfix in postfixes:
            postfix_length = len(postfix)
            if postfix == dst_name[-postfix_length:]:
                dst_name = dst_name[:-postfix_length]
                break

    for postfix in postfixes:
        src = osp.join(src_dir, src_name + postfix)
        if osp.exists(src):
            dst = osp.join(dst_folder, dst_name + postfix)
            if overwrite or not osp.exists(dst):
                if execute:
                    os.makedirs(dst_folder, exist_ok=True)
                    try:
                        if copy:
                            shutil.copy(src, dst)
                        else:
                            shutil.move(src, dst)
                    except Exception as e:
                        if not ignore_failed:
                            raise Exception(e)
                else:
                    print("[move_file_pairs]: ", src, dst)


def save_txt_jpg(path, image, content):
    """Save an image as a .png file and optionally save content as a .txt file.

    Args:
        path (str): The base file path.
        image (np.ndarray): The image to be saved.
        content (str, optional): The content to be written to a text file. Defaults to None.

    Returns:
        tuple: A tuple containing the paths to the saved image and text file.
    """
    # Determine the file extension
    file_extension = osp.splitext(path)[-1]

    # Create the .png file path
    png_path = path.replace(file_extension, '.png')

    os.makedirs(osp.dirname(png_path), exist_ok=True)
    # Save the image as a .png file
    cv2.imwrite(png_path, image)

    if content is None:
        return png_path, None

    # Create the .txt file path
    txt_path = path.replace(file_extension, '.txt')

    # Write the content to the .txt file
    with open(txt_path, 'w') as file:
        file.writelines(content)

    return png_path, txt_path


# ==============图像获取================


def plt2array():
    """Convert a Matplotlib plot to a NumPy array.

    Returns:
        np.ndarray: The RGBA image as a NumPy array.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # Convert the Matplotlib plot to a NumPy array
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()

    # Get the width and height of the canvas
    width, height = canvas.get_width_height()

    # Decode the string to get the ARGB image
    buffer = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

    # Reshape the buffer to (width, height, 4) for ARGB
    buffer.shape = (height, width, 4)

    # Convert ARGB to RGBA
    buffer = np.roll(buffer, 3, axis=2)

    # Create an Image object from the buffer
    image = Image.frombytes("RGBA", (width, height), buffer.tobytes())

    # Convert the Image object to a NumPy array
    plt.clf()
    return np.asarray(image)


def video2images(args):
    """Extract frames from videos and save them as images.

    Args:
        args (list): A list containing the thread index and a list of video paths.

    Example:
        if __name__ == "__main__":
            pattern = "xxx/*/*/*.mp4"
            all_data = glob.glob(pattern)
            print("total: ", len(all_data))
            cm.multi_process(video2images, all_data, num_threads=4)
    """
    thread_idx, all_videos = args
    print(f"{thread_idx} processing ", len(all_videos))

    for video in all_videos:
        folder = os.path.splitext(video)[0]
        saved = 0

        if os.path.exists(folder):
            saved = len(glob.glob(os.path.join(folder, "*.png")))
        else:
            os.makedirs(folder, exist_ok=True)

        cap = cv2.VideoCapture(video)
        index = 0
        ret, frame = cap.read()

        while ret:
            if index < saved:
                ret, frame = cap.read()
                index += 1
                continue

            if index % 1 == 0:
                new_path = os.path.join(folder, f"{str(index).zfill(5)}.png")
                cv2.imwrite(new_path, frame)

            index += 1
            ret, frame = cap.read()

        cap.release()

    print(f"{thread_idx} Finished!")


# ===============图像处理===============


def pad_image(img, target=None, border_type=cv2.BORDER_CONSTANT, value=(0, 0, 0), center=True, align=8):
    """Pad an image to a target size.

    Args:
        img (np.ndarray): The input image.
        target (tuple or int, optional): The target size. If not provided, the shorter side is padded to match the longer side.
        border_type (int, optional): The border type to use. Defaults to cv2.BORDER_CONSTANT.
        value (tuple, optional): The border color value. Defaults to (0, 0, 0).
        center (bool, optional): Whether to center the image. Defaults to True.
        align (int, optional): Alignment value for divisibility. Defaults to 8.

    Returns:
        tuple: The padded image, left padding, and top padding.
    """
    height, width = img.shape[:2]

    if target is None:
        target_height = target_width = max(height, width)
    else:
        if isinstance(target, int):
            target_height = target_width = target
        else:
            target_width, target_height = target
            assert target_width >= width, f"pad value too small: {[width, height]} -> {target}"
            assert target_height >= height, f"pad value too small: {[width, height]} -> {target}"

    target_height, target_width = (divisibility(x) for x in [target_height, target_width])

    top, left = 0, 0
    if center:
        top = max((target_height - height) // 2, 0)
        left = max((target_width - width) // 2, 0)

    bottom = max(target_height - height - top, 0)
    right = max(target_width - width - left, 0)

    if border_type == cv2.BORDER_CONSTANT:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, border_type, value=value)
    else:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=border_type)

    return img, left, top


def random_pad_image(image, target_size, border_type=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
    """Randomly pads an image to the target size.

    Args:
        image (numpy.ndarray): The input image to be padded.
        target_size (int or tuple): The target size for padding. If an integer is provided, both width and height will be set to this value. If a tuple is provided, it should be in the form (width, height).
        border_type (int, optional): Border type to be used for padding. Defaults to cv2.BORDER_CONSTANT.
        border_value (tuple, optional): Border color value for padding. Defaults to (0, 0, 0).

    Returns:
        tuple: The padded image and the x, y coordinates of the top-left corner of the original image within the padded image.
    """
    height, width = image.shape[:2]

    if isinstance(target_size, int):
        target_height = target_width = target_size
    else:
        target_width, target_height = target_size

    top, left = 0, 0
    bottom, right = max(target_height - height - top, 0), max(target_width - width - left, 0)

    if bottom - top > 1:
        top = np.random.randint(0, bottom)
        bottom -= top

    if right - left > 1:
        left = np.random.randint(0, right)
        right -= left

    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, borderType=border_type, value=border_value
    )

    padded_image, x_offset, y_offset = pad_image(
        padded_image, target_size, center=False, border_type=border_type, value=border_value
    )

    return padded_image, left + x_offset, top + y_offset


def size_pre_process(image, max_length=4096, **kwargs):
    """Pre-processes the size of an image based on various criteria.

    Args:
        image (numpy.ndarray): The input image to be resized.
        max_length (int, optional): The maximum allowed length for width or height. Defaults to 4096.
        **kwargs: Additional keyword arguments for resizing criteria.
            - interpolation (int, optional): Interpolation method for resizing. Defaults to None.
            - align (int, optional): Alignment value for divisibility. Defaults to 32.
            - hard (int or tuple, optional): Hard target size for resizing.
            - short (int, optional): Target size for the shorter dimension.
            - long (int, optional): Target size for the longer dimension.
            - height (int, optional): Target height for resizing.
            - width (int, optional): Target width for resizing.

    Returns:
        numpy.ndarray: The resized image.
    """
    align_function = partial(divisibility, r=kwargs.get('align', 32))
    height, width = image.shape[:2]

    if "hard" in kwargs:
        if isinstance(kwargs["hard"], int):
            target_width = target_height = align_function(kwargs["hard"])
        else:
            target_width, target_height = kwargs["hard"]
            target_width = align_function(target_width)
            target_height = align_function(target_height)
    elif "short" in kwargs:
        short_side = kwargs["short"]
        if height > width:
            target_width = short_side
            target_height = align_function(height / width * target_width)
        else:
            target_height = short_side
            target_width = align_function(width / height * target_height)
    elif "long" in kwargs:
        long_side = kwargs["long"]
        if height < width:
            target_width = long_side
            target_height = align_function(height / width * target_width)
        else:
            target_height = long_side
            target_width = align_function(width / height * target_height)
    elif "height" in kwargs:
        target_height = align_function(kwargs["height"])
        target_width = align_function(width / height * target_height)
    elif "width" in kwargs:
        target_width = align_function(kwargs["width"])
        target_height = align_function(height / width * target_width)
    else:
        return None

    if target_width > max_length:
        print(f"[size_pre_process] target_width({target_width}->{max_length})")
        target_width = max_length
    if target_height > max_length:
        print(f"[size_pre_process] target_height({target_height}->{max_length})")
        target_height = max_length

    interpolation_method = kwargs.get("interpolation", None)
    if interpolation_method is None:
        if target_width * target_height > height * width:
            interpolation_method = cv2.INTER_LINEAR
        else:
            interpolation_method = cv2.INTER_AREA

    return cv2.resize(image, (target_width, target_height), interpolation=interpolation_method)


def center_crop(image):
    """Crops the center of the image to create a square image.

    Args:
        image (numpy.ndarray): The input image to be cropped.

    Returns:
        tuple: The cropped image and the x, y coordinates of the top-left corner of the cropped area.
    """
    height, width = image.shape[:2]
    side_length = min(height, width)
    x_offset = int(np.ceil((width - side_length) // 2))
    y_offset = int(np.ceil((height - side_length) // 2))
    cropped_image = image[y_offset: y_offset + side_length, x_offset: x_offset + side_length, ...]
    return cropped_image, x_offset, y_offset


def cv_img_to_base64(image, quality=100):
    """Converts an OpenCV image to a base64 encoded string.

    Args:
        image (numpy.ndarray): The input image to be converted.
        quality (int, optional): The quality of the JPEG encoding. Defaults to 100.

    Returns:
        str: The base64 encoded string of the image.
    """
    img_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode(".jpg", image, img_params)
    base64_str = base64.b64encode(encoded_img).decode()
    return base64_str


def distance(point1, point2):
    """Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple): The first point (x, y).
        point2 (tuple): The second point (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def warp_regions(image, box):
    """Warps a region of the image defined by a quadrilateral box.

    Args:
        image (numpy.ndarray): The input image to be warped.
        box (list): A list of four points defining the quadrilateral (p0, p1, p2, p3).

    Returns:
        numpy.ndarray: The warped image region.
    """
    p0, p1, p2, p3 = box
    height = int(distance(p0, p3))
    width = int(distance(p0, p1))
    src_pts = np.float32([p0, p1, p3])
    dst_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    warped_image = cv2.warpAffine(image, affine_matrix, (width, height))
    return warped_image


def rotate_image(image, angle, center_point=None, scale=1.0, border_mode=cv2.BORDER_REPLICATE):
    """Rotates an image counterclockwise by a specified angle.

    Args:
        image (numpy.ndarray): The input image to be rotated.
        angle (float): The angle by which to rotate the image counterclockwise.
        center_point (tuple, optional): The point around which to rotate the image. Defaults to the center of the image.
        scale (float, optional): The scaling factor. Defaults to 1.0.
        border_mode (int, optional): Pixel extrapolation method. Defaults to cv2.BORDER_REPLICATE.

    Returns:
        numpy.ndarray: The rotated image.
    """
    height, width = image.shape[:2]
    if center_point is None:
        center_point = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=border_mode)
    return rotated_image


def rotate_location(angle, rect):
    """Rotates the coordinates of a rectangle by a given angle.

    Args:
        angle (float): The angle by which to rotate the rectangle, in degrees.
        rect (tuple): A tuple (x, y, width, height) representing the rectangle.

    Returns:
        list: A list of tuples representing the new coordinates of the rectangle's corners.
    """
    angle_radians = -angle * np.pi / 180.0
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    x, y, width, height = rect
    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0_new = (x0 - x) * cos_angle - (y0 - y) * sin_angle + x
    y0_new = (x0 - x) * sin_angle + (y0 - y) * cos_angle + y

    x1_new = (x1 - x) * cos_angle - (y1 - y) * sin_angle + x
    y1_new = (x1 - x) * sin_angle + (y1 - y) * cos_angle + y

    x2_new = (x2 - x) * cos_angle - (y2 - y) * sin_angle + x
    y2_new = (x2 - x) * sin_angle + (y2 - y) * cos_angle + y

    x3_new = (x3 - x) * cos_angle - (y3 - y) * sin_angle + x
    y3_new = (x3 - x) * sin_angle + (y3 - y) * cos_angle + y

    return [(x0_new, y0_new), (x1_new, y1_new), (x2_new, y2_new), (x3_new, y3_new)]


# =============图像debug=============
def has_chinese(text):
    """Checks if a string contains any Chinese characters.

    Args:
        text (str): The input string to be checked.

    Returns:
        bool: True if the string contains Chinese characters, False otherwise.
    """
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def put_text(
        image,
        text,
        position=None,
        background_color=None,
        text_color=None,
        text_size=None,
        chinese_font_path="resource/Songti.ttc",
):
    """Adds text to an image at a specified position with optional background color.

    Args:
        image (numpy.ndarray): The input image to which text will be added.
        text (str): The text to be added to the image.
        position (tuple, optional): The (x, y) coordinates for the text position. Defaults to (0, 0).
        background_color (tuple, optional): The background color for the text. Defaults to (0, 0, 0).
        text_color (tuple or int, optional): The color of the text. Defaults to 255 for grayscale images and (255, 255, 255) for color images.
        text_size (int, optional): The size of the text. Defaults to a value based on the image dimensions.
        chinese_font_path (str, optional): The path to the Chinese font file. Defaults to "resource/Songti.ttc".

    Returns:
        numpy.ndarray: The image with the added text.
    """
    text = str(text)
    is_gray = image.ndim == 2

    if position is None:
        position = (0, 0)
    if background_color is None:
        background_color = (0, 0, 0)
    if text_color is None:
        text_color = 255 if is_gray else (255, 255, 255)

    height, width = image.shape[:2]
    if text_size is None:  # base 30 for pillow equal 1 for opencv
        text_size = round(0.01 * np.sqrt(height ** 2 + width ** 2)) + 1

    has_chinese_char = has_chinese(text)

    # Convert image to contiguous array
    image = np.ascontiguousarray(image)
    if has_chinese_char:
        img_with_text = put_text_use_pillow(image, position, text, text_size, background_color, text_color,
                                            chinese_font_path)
    else:
        img_with_text = put_text_using_opencv(image, position, text, text_size, background_color, text_color)

    return img_with_text


def put_text_using_opencv(image, position, text, text_size, background_color, text_color):
    """Adds text to an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image to which text will be added.
        position (tuple): The (x, y) coordinates for the text position.
        text (str): The text to be added to the image.
        text_size (int): The size of the text.
        background_color (tuple or int): The background color for the text.
        text_color (tuple or int): The color of the text.

    Returns:
        numpy.ndarray: The image with the added text.
    """
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = text_size / 30
    thickness = 1

    # Offset position
    x1, y1 = np.array(position, dtype=int)
    x2, y2 = x1, y1
    font_sizes = []
    texts = text.replace('\r', '\n').split('\n')
    cur_texts = []
    (char_width, char_height), baseline = cv2.getTextSize('1234567890gj', font, font_scale, thickness)
    char_height = int(char_height + baseline)
    char_width = int(char_width / 12)

    for line in texts:
        cur_text = [line]
        while cur_text:
            line = cur_text.pop()
            if len(line) == 0:
                continue
            text_width, text_height = char_width * len(line), char_height
            if text_width > width and len(line) > 1:
                mid = max(int(width / char_width), 1)
                cur_text.append(line[mid:])
                cur_text.append(line[:mid])
            else:
                x2 = max(x2, x1 + text_width)
                y2 += text_height + 2
                font_sizes.append([text_width, text_height])
                cur_texts.append(line)

    x1, _ = get_offset_coordinates(x1, x2, 0, width)
    y1, _ = get_offset_coordinates(y1, y2, 0, height)

    for line, (tw, th) in zip(cur_texts, font_sizes):
        left_x, right_x = x1, x1 + tw
        top_y, bottom_y = y1, y1 + th
        if background_color != -1:
            try:
                cv2.rectangle(
                    image, (left_x - 1, top_y),
                    (right_x + 1, bottom_y),
                    background_color, -1, cv2.LINE_AA
                )
            except Exception as e:
                print(f"[put_text_using_opencv]: {e}, coordinates: {(left_x - 1, top_y), (right_x + 1, bottom_y)}")
        cv2.putText(
            image, line, (left_x, bottom_y - baseline),
            font, font_scale, text_color, thickness
        )
        x1, y1 = left_x, bottom_y + 1

    return image


def put_text_use_pillow(image, position, text, text_size, background_color, text_color, chinese_font_path):
    """Adds text to an image using Pillow, with support for Chinese characters.

    Args:
        image (numpy.ndarray): The input image to which text will be added.
        position (tuple): The (x, y) coordinates for the text position.
        text (str): The text to be added to the image.
        text_size (int): The size of the text.
        background_color (tuple or int): The background color for the text.
        text_color (tuple or int): The color of the text.
        chinese_font_path (str): The path to the Chinese font file.

    Returns:
        numpy.ndarray: The image with the added text.
    """
    text_size = max(text_size, 10)
    if osp.exists(chinese_font_path):
        font = ImageFont.truetype(chinese_font_path, int(text_size))
    else:
        chinese_font_path = fm.findfont(fm.FontProperties(family="AR PL UKai CN"))
        if osp.exists(chinese_font_path):
            font = ImageFont.truetype(chinese_font_path, int(text_size))
        else:
            print("[put_text_use_pillow]: 有中文, 但没有对应的字体.")
            font = None

    if font is None:
        return put_text_using_opencv(image, position, text, text_size, background_color, text_color)

    height, width = image.shape[:2]
    is_gray = image.ndim == 2

    # Offset position
    x1, y1 = np.array(position, dtype=int)
    x2, y2 = x1, y1
    font_sizes = []
    texts = text.replace('\r', '\n').split('\n')
    cur_texts = []

    for line in texts:
        cur_text = [line]
        while cur_text:
            line = cur_text.pop()
            if len(line) == 0:
                continue
            if pl_version < '9.5.0':  # 9.5.0 later
                left, top, right, bottom = font.getbbox(line)
                text_width, text_height = right - left, bottom - top
            else:
                text_width, text_height = font.getsize(line)
            if text_width > width and len(line) > 1:
                mid = max(int(width / (text_width / len(line))), 1)
                cur_text.append(line[mid:])
                cur_text.append(line[:mid])
            else:
                x2 = max(x2, x1 + text_width)
                y2 += text_height + 2
                font_sizes.append([text_width, text_height])
                cur_texts.append(line)

    x1, _ = get_offset_coordinates(x1, x2, 0, width)
    y1, _ = get_offset_coordinates(y1, y2, 0, height)

    if is_gray:
        img_pillow = Image.fromarray(image)
    else:
        img_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_pillow)

    for line, (tw, th) in zip(cur_texts, font_sizes):
        left_top_x, left_top_y = x1, y1
        right_bottom_x, right_bottom_y = x1 + tw, y1 + th
        if background_color != -1:
            cv2.rectangle(
                image, (left_top_x, left_top_y - 1),
                (right_bottom_x, right_bottom_y + 1),
                background_color, -1, cv2.LINE_AA
            )
        draw.text((left_top_x, left_top_y), line, font=font, fill=text_color)
        x1, y1 = left_top_x, right_bottom_y + 1

    image = np.asarray(img_pillow)
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def norm_for_show(array):
    """Normalizes an array for display purposes.

    Args:
        array (numpy.ndarray): The input array to be normalized.

    Returns:
        numpy.ndarray: The normalized array, scaled to the range [0, 255] and converted to uint8.
    """
    normalized_array = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
    return normalized_array


def imshow(window_name, image, original_size=False, delay=0, exit_key=113):
    """Displays an image in a window.

    Args:
        window_name (str): The name of the window.
        image (numpy.ndarray): The image to be displayed.
        original_size (bool, optional): Whether to display the image in its original size. Defaults to False.
        delay (int, optional): The delay in milliseconds for the waitKey function. Defaults to 0.
        exit_key (int, optional): The key code to exit the display. Defaults to 113 ('q').

    Returns:
        int: The key code pressed during the display.
    """
    if image is not None:
        if not original_size:
            height, width = image.shape[:2]
            if height > 2048 or width > 2048:
                image = size_pre_process(image, long=2048)
        cv2.imshow(window_name, image)
    key = cv2.waitKey(delay)
    if key == exit_key:
        exit()
    return key


def imwrite(file_path, image, overwrite=True):
    """Writes an image to a file.

    Args:
        file_path (str): The path to save the image.
        image (numpy.ndarray): The image to be saved.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to True.
    """
    if not overwrite and osp.exists(file_path):
        print(f"{file_path} already exists!")
        return
    os.makedirs(osp.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, image)


# ============标签处处理==============


def create_labelme_file(png_path, content=None, overwrite=False, labelme_version="5.0.1"):
    """Creates a LabelMe JSON file for the given PNG image.

    Args:
        png_path (str): Path to the PNG image file.
        content (dict, optional): Content to be written to the JSON file. Defaults to None.
        overwrite (bool, optional): Whether to overwrite the existing JSON file. Defaults to False.
        labelme_version (str, optional): Version of LabelMe. Defaults to "5.0.1".
    """
    json_path = osp.splitext(png_path)[0] + '.json'
    if osp.exists(json_path) and not overwrite:
        return

    # Create the content dictionary if not provided
    if content is None:
        content = create_labelme_content(
            None, png_path, [], labelme_version
        )

    # Write the content dictionary to a JSON file
    with open(json_path, 'w') as file_object:
        json.dump(content, file_object)


def create_labelme_content(img, png_path, shapes=None, labelme_version="5.0.1"):
    """Creates the content dictionary for a LabelMe JSON file.

    Args:
        img (numpy.ndarray or None): Image data. If None, the image will be read from png_path.
        png_path (str): Path to the PNG image file.
        shapes (list, optional): List of shapes to be included in the JSON file. Defaults to an empty list.
        labelme_version (str, optional): Version of LabelMe. Defaults to "5.0.1".

    Returns:
        dict: Content dictionary for the LabelMe JSON file.
    """
    if shapes is None:
        shapes = []

    # Convert the image to base64
    if img is None:
        img = cv2.imread(png_path)
    encoded_string = cv_img_to_base64(img)
    img_height, img_width = img.shape[:2]

    # Create the base_info dictionary
    base_info = {
        "version": labelme_version,
        "flags": {},
        "shapes": shapes,
        "imagePath": osp.basename(png_path),
        "imageData": encoded_string,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    return base_info


def create_labelme_shape(label: str, points, shape_type: str):
    """Creates a shape dictionary for a LabelMe JSON file.

    Args:
        label (str): Label for the shape.
        points (list): List of points defining the shape.
        shape_type (str): Type of the shape (e.g., "rectangle", "polygon").

    Returns:
        dict: Shape dictionary for the LabelMe JSON file.
    """
    points = np.reshape(points, [-1, 2]).squeeze().tolist()
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "flags": {}
    }


POLYGON_APPROX_TOLERANCE = 0.004


def compute_polygon_from_mask(mask, debug=False):
    """Extracts polygon contours from a binary mask image.

    Args:
        mask (numpy.ndarray): Binary mask image with values 0 or 1.
        debug (bool, optional): Whether to visualization. Defaults to False.

    Returns:
        list: List of polygons, where each polygon is represented as an array of points.
    """
    import skimage.measure
    # Pad the mask to ensure contours are detected at the edges
    padded_mask = np.pad(mask, pad_width=1)
    contours = skimage.measure.find_contours(padded_mask, level=0.5)

    if len(contours) == 0:
        print("No contour found, returning empty polygon.")
        return []

    polygons = []

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        # Approximate the polygon
        polygon = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
        )
        # Clip the polygon to the mask dimensions
        polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
        # Remove the last point if it is a duplicate of the first point
        polygon = polygon[:-1]

        # Optional visualization (disabled by default)
        if debug:
            vision = (255 * np.stack([mask] * 3, axis=-1)).astype(np.uint8)
            for y, x in polygon.astype(int):
                cv2.circle(vision, (x, y), 3, (0, 0, 222), -1)
            cv2.imshow("Polygon", vision)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Append the polygon with coordinates in (x, y) format
        polygons.append(polygon[:, ::-1])

    return polygons


class LabelObject(object):
    """Class representing a labeled object with various attributes."""

    def __init__(self):
        self.type = None
        self.pts = None
        self.ori_pts = None
        self.pts_normed = None
        self.label = None
        self.box = None
        self.height = None
        self.width = None

    def __str__(self):
        return f"type: {self.type}, label: {self.label}"


def parse_json(path, polygon, return_dict=False) -> [list, np.ndarray, str]:
    """Parses a JSON file and extracts image and shape information.

    Args:
        path (str): Path to the JSON file.
        polygon (bool): Whether to convert points to polygon format.
        return_dict (bool, optional): Whether to return a dictionary of objects. Defaults to False.

    Returns:
        tuple: A tuple containing a list or dictionary of LabelObject instances, the image, and the basename.
    """
    assert path.endswith('.json')
    info = json.load(open(path, "r"))
    base64_str = info.get("imageData", None)
    if base64_str is None:
        img = cv2.imread(path.replace(".json", ".png"))
    else:
        img_str = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        image_height = info.get("imageHeight", None)
        image_width = info.get("imageWidth", None)
    else:
        image_height, image_width = img.shape[:2]

    obj_list = []
    for shape in info.get("shapes", []):
        obj = LabelObject()
        obj.label = shape.get("label", None)
        pts = shape.get("points", [])
        obj.ori_pts = np.reshape(pts, (-1, 2)).astype(int)
        if polygon and len(pts) == 2:
            x1, y1, x2, y2 = np.array(pts).flatten()
            pts = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
        if not polygon and len(pts) > 3:
            pts = get_min_rect(pts).flatten()[:4]
        obj.pts = np.reshape(pts, (-1, 2))
        obj.type = shape.get("shape_type", "")
        obj.height = image_height
        obj.width = image_width
        obj_list.append(obj)
        # =====processed=======
        obj.pts_normed = np.reshape(obj.pts, [-1, 2]) / np.array(
            [image_width, image_height]
        )
    basename = osp.basename(path).split(".")[0]
    if return_dict:
        obj_dict = collections.defaultdict(list)
        for obj in obj_list:
            obj_dict[obj.label].append(obj)
        return obj_dict, img, basename
    return obj_list, img, basename


def show_yolo_label(img, lines, xywh=True, classes: dict = None, colors=None, thickness=2):
    """Displays YOLO labels on an image.

    Args:
        img (numpy.ndarray): The image on which to display the labels.
        lines (list): List of label lines, each containing class index and bounding box coordinates.
        xywh (bool, optional): Whether the bounding box coordinates are in (x, y, width, height) format. Defaults to True.
        classes (dict, optional): Dictionary mapping class indices to class names. Defaults to None.
        colors (list, optional): List of colors for each class. Defaults to None.
        thickness (int, optional): Thickness of the bounding box lines. Defaults to 2.

    Returns:
        tuple: The image with labels and a list of points.
    """
    if classes is None:
        classes = {i: i for i in range(10)}
    if colors is None:
        colors = create_color_list(len(classes))
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    pts = []
    for line in lines:
        if not line: continue
        sp = line.strip().split(" ")
        idx, a, b, c, d = [float(x) for x in sp]
        idx = int(idx)
        if xywh:
            x1, y1, x2, y2 = (
                    xywh2xyxy([a, b, c, d]) * np.array([width, height, width, height])
            ).astype(int)
        else:
            x1, y1, x2, y2 = (
                    np.array([a, b, c, d]) * np.array([width, height, width, height])
            ).astype(int)

        if thickness == -1:
            mask = cv2.rectangle(mask, (x1, y1), (x2, y2), colors[idx], thickness)
        else:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[idx], thickness)
        img = put_text(img, str(classes[idx]), (x1, y1), (0, 0, 0), (222, 222, 222))
        pts.append([idx, x1, y1, x2, y2])
    if thickness == -1:
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 1)
    return img, pts


def show_yolo_file(jpg_path, xywh=True, classes=None, colors=None, thickness=2):
    """Displays YOLO labels on an image from a file.

    Args:
        jpg_path (str): Path to the JPEG image file.
        xywh (bool, optional): Whether the bounding box coordinates are in (x, y, width, height) format. Defaults to True.
        classes (dict, optional): Dictionary mapping class indices to class names. Defaults to None.
        colors (list, optional): List of colors for each class. Defaults to None.
        thickness (int, optional): Thickness of the bounding box lines. Defaults to 2.

    Returns:
        tuple: The image with labels and a list of points.
    """
    img = cv2.imread(jpg_path)
    txt = osp.splitext(jpg_path)[0] + ".txt"
    with open(txt, "r") as fo:
        lines = fo.readlines()
    img, pts = show_yolo_label(img, lines, xywh, classes, colors, thickness)
    img = put_text(img, osp.basename(jpg_path))
    return img, pts


# =========Warp face from insightface=======


def estimate_norm(landmarks, image_size=112):
    """Estimates the normalization transformation matrix for given landmarks.

    Args:
        landmarks (numpy.ndarray): Array of shape (5, 2) containing the facial landmarks.
        image_size (int, optional): Size of the output image. Defaults to 112.

    Returns:
        numpy.ndarray: The 2x3 transformation matrix.
    """
    from skimage import transform as trans
    arcface_dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    assert landmarks.shape == (5, 2), "Landmarks should be of shape (5, 2)."
    assert image_size % 112 == 0 or image_size % 128 == 0, "Image size should be a multiple of 112 or 128."

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    transformation_matrix = tform.params[0:2, :]

    return transformation_matrix


def norm_crop(img, landmarks, image_size=112):
    """Normalizes and crops an image based on facial landmarks.

    Args:
        img (numpy.ndarray): The input image.
        landmarks (numpy.ndarray): Array of shape (5, 2) containing the facial landmarks.
        image_size (int, optional): Size of the output image. Defaults to 112.

    Returns:
        tuple: The warped image and the transformation matrix.
    """
    transformation_matrix = estimate_norm(landmarks, image_size)
    warped_image = cv2.warpAffine(img, transformation_matrix, (image_size, image_size), borderValue=0.0)

    return warped_image, transformation_matrix


def warp_face(img, x1, y1, x2, y2):
    """Warps a face in an image based on bounding box coordinates.

    Args:
        img (numpy.ndarray): The input image.
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
        numpy.ndarray: The warped image.
    """
    center_x, center_y, face_width, face_height = xyxy2xywh([x1, y1, x2, y2]).flatten().astype(int)
    scale = 256 / (max(face_width, face_height) * 1.5)
    return transform(img, (center_x, center_y), 256, scale, 0)


def transform(data, center, output_size, scale, rotation):
    """Applies a series of transformations to the input data.

    Args:
        data (numpy.ndarray): The input image data.
        center (tuple): The center point for the transformation.
        output_size (int): The size of the output image.
        scale (float): The scaling factor.
        rotation (float): The rotation angle in degrees.

    Returns:
        tuple: The transformed image and the transformation matrix.
    """
    from skimage import transform as trans
    scale_ratio = scale
    rotation_radians = float(rotation) * np.pi / 180.0

    # Define the series of transformations
    scale_transform = trans.SimilarityTransform(scale=scale_ratio)
    translation_transform1 = trans.SimilarityTransform(translation=(-center[0] * scale_ratio, -center[1] * scale_ratio))
    rotation_transform = trans.SimilarityTransform(rotation=rotation_radians)
    translation_transform2 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))

    # Combine the transformations
    combined_transform = scale_transform + translation_transform1 + rotation_transform + translation_transform2
    transformation_matrix = combined_transform.params[0:2]

    # Apply the transformation
    cropped_image = cv2.warpAffine(data, transformation_matrix, (output_size, output_size), borderValue=0.0)

    return cropped_image, transformation_matrix


def transform_points_2d(points, transformation_matrix):
    """Transforms 2D points using a given transformation matrix.

    Args:
        points (numpy.ndarray): Array of shape (N, 2) containing the 2D points.
        transformation_matrix (numpy.ndarray): The 2x3 transformation matrix.

    Returns:
        numpy.ndarray: Array of transformed 2D points.
    """
    transformed_points = np.zeros(shape=points.shape, dtype=np.float32)
    for i in range(points.shape[0]):
        point = points[i]
        homogeneous_point = np.array([point[0], point[1], 1.0], dtype=np.float32)
        transformed_point = np.dot(transformation_matrix, homogeneous_point)
        transformed_points[i] = transformed_point[0:2]

    return transformed_points


def transform_points_3d(points, transformation_matrix):
    """Transforms 3D points using a given transformation matrix.

    Args:
        points (numpy.ndarray): Array of shape (N, 3) containing the 3D points.
        transformation_matrix (numpy.ndarray): The 2x3 transformation matrix.

    Returns:
        numpy.ndarray: Array of transformed 3D points.
    """
    scale = np.sqrt(transformation_matrix[0][0] ** 2 + transformation_matrix[0][1] ** 2)
    transformed_points = np.zeros(shape=points.shape, dtype=np.float32)
    for i in range(points.shape[0]):
        point = points[i]
        homogeneous_point = np.array([point[0], point[1], 1.0], dtype=np.float32)
        transformed_point = np.dot(transformation_matrix, homogeneous_point)
        transformed_points[i][0:2] = transformed_point[0:2]
        transformed_points[i][2] = points[i][2] * scale

    return transformed_points


# =======3D==========
def pixel_to_camera_3d(pt_2d, depth, camera_matrix):
    """Converts 2D pixel coordinates to 3D camera coordinates.

    Args:
        pt_2d (tuple): The (u, v) coordinates in the distorted image.
        depth (float or numpy.ndarray): The depth value or depth map.
        camera_matrix (numpy.ndarray): The camera intrinsic matrix.

    Returns:
        tuple: The 3D camera coordinates and the depth value.
    """
    if np.ndim(depth) > 1:
        x, y = map(int, pt_2d)
        depth = depth[y, x]

    if depth == 0:
        return None, None

    u_distorted, v_distorted = pt_2d
    homogeneous_uv1 = np.array([u_distorted, v_distorted, 1])
    camera_xy1 = np.linalg.inv(camera_matrix) @ homogeneous_uv1
    camera_xyz = camera_xy1 * depth

    return camera_xyz, depth


def pixel_to_camera_3d_numpy(p_uv, depth, camera_matrix):
    """Converts 2D pixel coordinates to 3D camera coordinates using NumPy.

    Args:
        p_uv (numpy.ndarray): Array of shape (N, 2) containing the 2D pixel coordinates.
        depth (numpy.ndarray or float): Array of shape (N,) containing the depth values or a single depth value.
        camera_matrix (numpy.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        tuple: The 3D camera coordinates (N, 3) and the depth values (N,).
    """
    if np.ndim(depth) == 2:
        indices = np.reshape(p_uv, (-1, 2)).astype(int)
        depth = depth[indices[:, 1], indices[:, 0]]

    u_distorted, v_distorted = np.split(p_uv, 2, axis=-1)
    homogeneous_uv1 = np.stack([u_distorted, v_distorted, np.ones_like(u_distorted)], axis=1)  # [N, 3, 1]
    camera_xy1 = np.matmul(np.linalg.inv(camera_matrix), homogeneous_uv1)[..., 0]  # [N, 3, 1]
    camera_xyz = camera_xy1 * depth[:, None]  # depth shape: (N,)

    return camera_xyz, depth


def draw_gaze(image, start, pitch_yaw, length, thickness=1, color=(0, 0, 255), is_degree=False):
    """Draws the gaze angle on the given image based on eye positions.

    Args:
        image (numpy.ndarray): The input image.
        start (tuple): The starting (x, y) coordinates for the gaze line.
        pitch_yaw (tuple): The pitch and yaw angles.
        length (int): The length of the gaze line.
        thickness (int, optional): The thickness of the gaze line. Defaults to 1.
        color (tuple, optional): The color of the gaze line in BGR format. Defaults to (0, 0, 255).
        is_degree (bool, optional): Whether the pitch and yaw are in degrees. Defaults to False.

    Returns:
        tuple: The image with the gaze line drawn and the angle in degrees.
    """
    if is_degree:
        pitch_yaw = np.deg2rad(pitch_yaw)

    pitch, yaw = pitch_yaw
    x, y = start

    if np.ndim(image) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    dx = length * np.cos(pitch) * np.sin(yaw)
    dy = -length * np.sin(pitch)

    cv2.arrowedLine(
        image,
        (int(x), int(y)),
        (int(x + dx), int(y + dy)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )

    angle = np.rad2deg(np.arctan2(dy, dx))

    return image, angle


def gaze_3d_to_2d(gaze_3d, transformation_matrix=None):
    """Converts 3D gaze vector to 2D pitch and yaw angles.

    Args:
        gaze_3d (numpy.ndarray): The 3D gaze vector.
        transformation_matrix (numpy.ndarray, optional): The transformation matrix. Defaults to None.

    Returns:
        tuple: The pitch and yaw angles in degrees.
    """
    if transformation_matrix is not None:
        gaze_3d = np.dot(transformation_matrix, gaze_3d)

    gaze_3d = gaze_3d / np.linalg.norm(gaze_3d)
    dx, dy, dz = gaze_3d
    pitch = np.rad2deg(np.arcsin(-dy))  # -dy: Up is positive
    yaw = np.rad2deg(np.arctan(-dx / (dz + 1e-7)))  # -dx: Left is positive

    return pitch, yaw


def gaze_2d_to_3d(pitch, yaw, is_degree=True):
    """Converts 2D pitch and yaw angles to a 3D gaze vector.

    Args:
        pitch (float or numpy.ndarray): The pitch angle.
        yaw (float or numpy.ndarray): The yaw angle.
        is_degree (bool, optional): Whether the angles are in degrees. Defaults to True.

    Returns:
        numpy.ndarray: The 3D gaze vector.
    """
    if is_degree:
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    pitch = np.reshape(pitch, (-1, 1))
    yaw = np.reshape(yaw, (-1, 1))
    batch_size = np.shape(pitch)[0]
    gaze = np.zeros((batch_size, 3))
    gaze[:, 0] = np.cos(pitch) * np.sin(yaw)
    gaze[:, 1] = -np.sin(pitch)
    gaze[:, 2] = -np.cos(pitch) * np.cos(yaw)
    gaze = gaze / np.linalg.norm(gaze, axis=1, keepdims=True)

    return gaze


def cosine_similarity_deg(a, b):
    """Calculates the cosine similarity between two vectors and returns the angle in degrees.

    Args:
        a (numpy.ndarray): First input vector of shape (N, D).
        b (numpy.ndarray): Second input vector of shape (N, D).

    Returns:
        numpy.ndarray: Array of angles in degrees between the input vectors.
    """
    a_normalized = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_normalized = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot_product = np.sum(a_normalized * b_normalized, axis=1)
    dot_product = np.clip(dot_product, a_min=-1.0, a_max=0.999999)
    angle_rad = np.arccos(dot_product)  # radians
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg


def compute_euler(rotation_vector, translation_vector):
    """Computes Euler angles from a rotation vector.

    Args:
        rotation_vector (numpy.ndarray): The rotation vector.
        translation_vector (numpy.ndarray): The translation vector.

    Returns:
        numpy.ndarray: The Euler angles (pitch, yaw, roll) in degrees.
    """
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch = euler_angles[0]
    yaw = euler_angles[1]
    roll = euler_angles[2]
    rotation_params = np.array([pitch, yaw, roll]).flatten()

    return rotation_params


class NormalWarp:
    def __init__(self, camera_matrix, distortion_coeffs, distance_norm, focal_norm):
        """Initializes the NormalWarp class.

        Args:
            camera_matrix (numpy.ndarray): The camera intrinsic matrix.
            distortion_coeffs (numpy.ndarray): The camera distortion coefficients.
            distance_norm (float): Normalized distance between eye and camera.
            focal_norm (float): Focal length of the normalized camera.
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.face_points = np.array([
            [-45.0968, -21.3129, 21.3129, 45.0968, -26.2996, 26.2996],
            [-0.4838, 0.4838, 0.4838, -0.4838, 68.595, 68.595],
            [2.397, -2.397, -2.397, 2.397, -0.0, -0.0]
        ])
        self.face_points_t = self.face_points.T.reshape(-1, 1, 3)

        self.distance_norm = distance_norm
        self.roi_size = (448, 448)
        self.normalized_camera_matrix = np.array([
            [focal_norm, 0, self.roi_size[0] / 2],
            [0, focal_norm, self.roi_size[1] / 2],
            [0, 0, 1.0],
        ])

    def estimate_head_pose(self, landmarks, iterate=True):
        """Estimates the head pose from facial landmarks.

        Args:
            landmarks (numpy.ndarray): Array of shape (N, 2) containing the facial landmarks.
            iterate (bool, optional): Whether to further optimize the pose estimation. Defaults to True.

        Returns:
            tuple: Rotation vector, translation vector, and Euler angles.
        """
        landmarks = np.reshape(landmarks, (-1, 2))
        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_points_t, landmarks, self.camera_matrix, self.distortion_coeffs, flags=cv2.SOLVEPNP_EPNP
        )

        if iterate:
            _, rotation_vector, translation_vector = cv2.solvePnP(
                self.face_points_t, landmarks, self.camera_matrix, self.distortion_coeffs, rotation_vector,
                translation_vector, True
            )
        head_euler = compute_euler(rotation_vector, translation_vector)
        return rotation_vector, translation_vector, head_euler

    def __call__(self, image, landmarks):
        """Normalizes and warps the face in the image based on facial landmarks.

        Args:
            image (numpy.ndarray): The input image.
            landmarks (numpy.ndarray): Array of shape (N, 2) containing the facial landmarks.

        Returns:
            tuple: The warped face image, rotation matrix, and warp matrix.
        """
        rotation_vector, translation_vector, _ = self.estimate_head_pose(landmarks)

        translation_vector = np.repeat(translation_vector, 6, axis=1)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        face_center_3d = np.dot(rotation_matrix, self.face_points) + translation_vector
        face_center = np.sum(face_center_3d, axis=1, dtype=np.float32) / 6.0

        distance = np.linalg.norm(face_center)
        face_center /= distance
        forward = face_center.reshape(3)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        right = rotation_matrix[:, 0]
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        rotation_matrix = np.c_[right, down, forward].T

        scale_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, self.distance_norm / distance],
        ])
        warp_matrix = np.dot(
            np.dot(self.normalized_camera_matrix, scale_matrix),
            np.dot(rotation_matrix, self.camera_matrix_inv)
        )

        face_image = cv2.warpPerspective(image, warp_matrix, self.roi_size)
        return face_image, rotation_matrix, warp_matrix


# =====================deprecated==========

def deprecated(func):
    """
    这是一个装饰器，用于标记函数为已弃用。当使用该函数时，会发出警告。

    Args:
        func (function): 被装饰的函数。

    Returns:
        function: 包装后的新函数。
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # 关闭过滤器
        warnings.warn(
            f"调用已弃用的函数 {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2
        )
        warnings.simplefilter('default', DeprecationWarning)  # 恢复过滤器
        return func(*args, **kwargs)

    return new_func

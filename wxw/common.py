import collections
import os
import pdb
import glob
import cv2
import time
import json
import random
import base64
import shutil
import hashlib
import numpy as np
import os.path as osp
from io import BytesIO
from functools import partial
import matplotlib.pylab as plt
from multiprocessing import Pool
from collections import defaultdict

import argparse
import warnings
import functools
import matplotlib.font_manager as fm
from PIL import __version__ as pl_version
from PIL import Image, ImageFont, ImageDraw

np.random.seed(123456)
random.seed(123456)

# Terminal多彩输出
print_red = lambda x: print(f"\033[31m{x}\033[0m")
print_green = lambda x: print(f"\033[32m{x}\033[0m")
print_yellow = lambda x: print(f"\033[32m{x}\033[0m")
print_blue = lambda x: print(f"\033[33m{x}\033[0m")


def deprecated(func):
    """这是一个装饰器，用于标记函数为已弃用。当使用该函数时，会发出警告。"""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # 关闭过滤器
        warnings.warn(f"调用已弃用的函数 {func.__name__}.", category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # 恢复过滤器
        return func(*args, **kwargs)

    return new_func


# ========inference=====
class ONNXRunner:
    def __init__(self, path):
        import onnxruntime

        providers = [
            "CUDAExecutionProvider", 'CoreMLExecutionProvider', 'CPUExecutionProvider'
        ]
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        print("inputs: ", [x.name for x in self.session.get_inputs()])
        print("outputs: ", [x.name for x in self.session.get_outputs()])

    def __call__(self, img):
        try:
            return self.session.run(
                [x.name for x in self.session.get_outputs()],
                {self.session.get_inputs()[0].name: img},
            )
        except Exception as e:
            print("[ONNXRunner] why?: ")
            print(e)
            print(self.session.get_inputs()[0])
            print(img.shape)


# =============基础方法===============
def update_args(old_, new_) -> argparse.Namespace:
    import yaml
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


def safe_replace(src, _old, _new):
    dst = src.replace(_old, _new)
    if dst == src:
        print("no replace!")
        return None
    return dst


def deprecated(func):
    """这是一个装饰器，用于标记函数为已弃用。当使用该函数时，会发出警告。"""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # 关闭过滤器
        warnings.warn(f"调用已弃用的函数 {func.__name__}.", category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # 恢复过滤器
        return func(*args, **kwargs)

    return new_func


def md5sum(file_path):
    with open(file_path, "rb") as file:
        md5_hash = hashlib.md5()
        while True:
            data = file.read(4096)  # 每次读取4KB数据
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()


def print_format(string, a, func, b):
    format = f"{a:<5.3f} {func} {b:<5.3f}"
    if func == '/':
        b = b + 1e-4
    c = eval(f"{a} {func} {b}")
    print(f"{string:<20}: {format} = {c:.3f}")
    return c


def divisibility(a, r=32):
    """整除"""
    if r == 1:
        return int(a)
    return int(np.ceil(a / r) * r)


def get_offset_coordinates(start_point, end_point, min_value, max_value):
    """
    Adjusts the start and end points of a line segment to ensure they fall within the specified range.
    If the length of the line segment is greater than the range, a warning is printed and the original points are returned.

    Parameters:
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

    end_offset = max(0, min_value - start_point)
    start_point = max(min_value, start_point)
    start_offset = max(0, end_point - max_value)
    end_point = min(max_value, end_point)
    start_point = max(start_point - start_offset, min_value)
    end_point = min(end_point + end_offset, max_value)

    return start_point, end_point


def cost_time(func):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(
            f"[INFO] [{func.__name__}] coast time:{(time.perf_counter() - t) * 1000:.4f}ms"
        )
        return result

    return wrapper


def xywh2xyxy(pts):
    pts = np.reshape(pts, [-1, 4])
    cx, cy, w, h = np.split(pts, 4, 1)
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    res = np.concatenate([x1, y1, x2, y2], axis=1)
    res = np.clip(res, 0, np.inf)
    # return res[0] if pts.shape[0] == 1 else res
    return np.squeeze(res)


def xyxy2xywh(pts):
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
    pts:
     [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
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


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def random_color(amin, amax):
    b = np.random.randint(amin, amax)
    g = np.random.randint(amin, amax)
    r = np.random.randint(amin, amax)
    return b, g, r


def create_color_lst(number):
    if number < 10:
        colors = np.array(plt.cm.tab10.colors)
    else:
        colors = np.array(plt.cm.tab20.colors)
    colors = (colors[:number - 1, ::-1] * 255).astype(np.uint8)
    colors = np.insert(colors, 0, (0, 0, 0))
    return colors


def make_color_table(number):
    if number < 10:
        colors = np.array(plt.cm.tab10.colors[:number]) * 255
        colors = colors[:, ::-1].astype(np.uint8)
    else:
        colors = np.array(plt.cm.tab20.colors[:number]) * 255
        colors = colors[:, ::-1].astype(np.uint8)
    color_table = {}
    for i in range(number):
        color_table[i] = tuple(colors[i].tolist())
    for i in range(number - len(color_table), number):
        color_table[i] = random_color(127, 255)
    return color_table


def multi_process(process_method, need_process_data, num_thread=1):
    """
    'process_method' should be:
    def process_method(args):
        thread_idx, need_process_data = args
        .....
    """
    if num_thread == 1:
        process_method([0, need_process_data])
    else:
        begin = 0
        total = len(need_process_data)
        interval = int(np.ceil(total / num_thread))
        end = interval

        works = []
        index = 0
        while begin < total:
            index += 1
            works.append([index, need_process_data[begin:end]])
            begin += interval
            end += interval
        pool = Pool(num_thread)
        pool.map(process_method, works)


def simplify_number(decimal):
    from fractions import Fraction

    # 使用Fraction类将小数转换为最简分数
    fraction = Fraction(decimal).limit_denominator()
    # 打印最简分数形式
    # print(f"小数：{decimal}")
    # print(f"最简分数：{fraction.numerator}/{fraction.denominator}")
    string = f"{fraction.numerator}/{fraction.denominator}"
    return string, fraction.numerator, fraction.denominator


# =========Files: 文件移动和写入============
def merge_path(path, flag):
    sp = path.split(os.sep)
    idx = sp.index(flag)
    return '_'.join(sp[idx:])


def move_file_pairs(
        path,
        dst_folder,
        dst_name=None,
        postfixes=None,
        copy=True,
        do=False,
        empty_undo=False,
        empty_del=False
):
    if empty_undo and os.path.getsize(path) == 0:
        if empty_del: os.remove(path)
        return
    prefix, self_post = osp.splitext(path)
    if postfixes is None:
        postfixes = [self_post]
    src_dir = osp.dirname(prefix)
    src_name = osp.basename(prefix)
    if dst_name is None:
        dst_name = src_name
    else:
        for postfix in postfixes:
            p_l_ = len(postfix)
            if postfix == dst_name[-p_l_:]:
                dst_name = dst_name[:-p_l_]
                break
    for postfix in postfixes:
        src = osp.join(src_dir, src_name + postfix)
        if osp.exists(src):
            dst = osp.join(dst_folder, dst_name + postfix)
            if not osp.exists(dst):
                if do:
                    os.makedirs(dst_folder, exist_ok=True)
                    if copy:
                        shutil.copy(src, dst)
                    else:
                        shutil.move(src, dst)
                else:
                    print("[move_file_pairs]: ", src, dst)


def save_txt_jpg(path, image, content):
    post_fix = osp.splitext(path)[-1]
    jpg_path = path.replace(post_fix, '.png')
    imwrite(jpg_path, image)
    if content is None:
        return jpg_path, None
    txt_path = path.replace(post_fix, '.txt')
    with open(txt_path, 'w') as fo:
        fo.writelines(content)
    return jpg_path, txt_path


# ==============图像获取================


def plt2array():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    plt.clf()
    return np.asarray(image)


def download_image_from_url(image_url):
    import requests
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return img


def read_avif_img(path):
    AVIFimg = Image.open(path)
    img = np.array(AVIFimg)
    return img[..., ::-1]


def video2images(args):
    """
    args: [线程id: int && 全部路径: list]

    if __name__ == "__main__":
        pattern = "xxx/*/*/*.mp4"
        all_data = glob.glob(pattern)
        print("total: ", len(all_data))
        cm.multi_process(v2f, all_data, num_thread=4)
    """
    idx, all_video = args
    print(f"{idx} processing ", len(all_video))
    for video in all_video:
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
    print(f"{idx} Finished!")


# ===============图像处理===============


def pad_image(img, target=None, board_type=cv2.BORDER_CONSTANT, value=(0, 0, 0), centre=True):
    """

    Args:
        img:
        target:  如何没有提供target，那么短边被填充到和长边一样的尺寸
        board_type:
        value:
        centre:

    Returns:

    """
    height, width = img.shape[:2]
    if target is None:
        t_h = t_w = max(height, width)
    else:
        if isinstance(target, int):
            t_h = t_w = target
        else:
            t_w, t_h = target
    top, left = 0, 0
    if centre:
        top = max((t_h - height) // 2, 0)
        left = max((t_w - width) // 2, 0)
    bottom, right = max(t_h - height - top, 0), max(t_w - width - left, 0)
    if board_type == cv2.BORDER_CONSTANT:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
        )
    else:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=board_type
        )
    return img, left, top


def random_pad_image(img, target, board_type=cv2.BORDER_CONSTANT, value=(0, 0, 0)):
    """"""
    height, width = img.shape[:2]
    if isinstance(target, int):
        t_h = t_w = target
    else:
        t_w, t_h = target
    top, left = 0, 0
    bottom, right = max(t_h - height - top, 0), max(t_w - width - left, 0)
    if bottom - top > 1:
        top = np.random.randint(0, bottom)
        bottom -= top
    if right - left > 1:
        left = np.random.randint(0, right)
        right -= left
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=board_type
    )
    img, x, y = pad_image(img, target, centre=False, board_type=board_type, value=value)
    return img, left + x, top + y


def size_pre_process(img, longest=4096, **kwargs):
    """kwargs
    interpolation

    """
    align_fun = partial(divisibility, r=kwargs.get('align', 32))
    h, w = img.shape[:2]
    if "hard" in kwargs:
        if isinstance(kwargs["hard"], int):
            rw = rh = align_fun(kwargs["hard"])
        else:
            rw, rh = kwargs["hard"]
            rw = align_fun(rw)
            rh = align_fun(rh)
    elif "short" in kwargs:
        short = kwargs["short"]
        if h > w:
            rw = short
            rh = align_fun(h / w * rw)
        else:
            rh = short
            rw = align_fun(w / h * rh)
    elif "long" in kwargs:
        long = kwargs["long"]
        if h < w:
            rw = long
            rh = align_fun(h / w * rw)
        else:
            rh = long
            rw = align_fun(w / h * rh)
    elif "height" in kwargs:
        rh = align_fun(kwargs["height"])
        rw = align_fun(w / h * rh)
    elif "width" in kwargs:
        rw = align_fun(kwargs["width"])
        rh = align_fun(h / w * rw)
    else:
        return None
    if rw > longest:
        print(f"[size_pre_process] rw({rw}->{longest})")
        rw = longest
    if rh > longest:
        print(f"[size_pre_process] rh({rh}->{longest})")
        rh = longest
    interpolation = kwargs.get("interpolation", None)
    if interpolation is None:
        if rw * rh > h * w:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
    return cv2.resize(img, (rw, rh), interpolation=interpolation)


def center_crop(image):
    height, width = image.shape[:2]
    side = min(height, width)
    x = int(np.ceil((width - side) // 2))
    y = int(np.ceil((height - side) // 2))
    image = image[y: y + side, x: x + side, ...]
    return image, x, y


def cv_img_to_base64(img, quality=100):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    base64_str = cv2.imencode(".jpg", img, img_param)[1]
    return base64.b64encode(base64_str).decode()


def warp_regions(img, box):
    p0, p1, p2, p3 = box
    h = int(distance(p0, p3))
    w = int(distance(p0, p1))
    pts1 = np.float32([p0, p1, p3])
    pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    return cv2.warpAffine(img, cv2.getAffineTransform(pts1, pts2), (w, h))


def rotate_image(img, angle, point=None, scale=1.0, borderMode=cv2.BORDER_REPLICATE):
    """逆时针旋转为正"""
    height, width = img.shape[:2]
    if point is None:
        point = (width // 2, height // 2)
    rotate_mtx = cv2.getRotationMatrix2D(point, angle, scale)
    return cv2.warpAffine(img, rotate_mtx, (width, height), borderMode=borderMode)


def rotate_location(angle, rect):
    """
    rect: x,y,w,h
    """
    anglePi = -angle * np.pi / 180.0
    cosA = np.cos(anglePi)
    sinA = np.sin(anglePi)

    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    return [(x0n, y0n), (x1n, y1n), (x2n, y2n), (x3n, y3n)]


def letterbox(
        im,
        new_shape=640,
        color=(114, 114, 114),
        auto=True,
        scale_fill=False,
        scale_up=True,
        stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


# =============图像debug=============
def put_text(
        im0,
        text,
        pts=None,
        bg_color=None,
        text_color=None,
        tl=None,
        zh_font_path="resource/Songti.ttc",
):
    text = str(text)
    # ============config========
    is_gray = im0.ndim == 2
    if pts is None:
        pts = (0, 0)
    if bg_color is None:
        bg_color = (0, 0, 0)
    if text_color is None:
        text_color = 255 if is_gray else (255, 255, 255)
    height, width = im0.shape[:2]
    if tl is None:  # base 30 for pillow equal 1 for opencv
        tl = round(0.01 * np.sqrt(height ** 2 + width ** 2)) + 1
    has_chinese_char = has_chinese(text)

    # ==========write===========
    im0 = np.ascontiguousarray(im0)
    if has_chinese_char:
        img = put_text_use_pillow(im0, pts, text, tl, bg_color, text_color, zh_font_path)
    else:
        img = put_text_using_opencv(im0, pts, text, tl, bg_color, text_color)
    return img


def has_chinese(string):
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def put_text_using_opencv(img, pts, text, tl, bg_color, text_color):
    img = np.ascontiguousarray(img)
    height, width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = tl / 30
    thickness = 1
    # ==========offset position=======
    x1, y1 = np.array(pts, dtype=int)
    x2, y2 = x1, y1
    font_sizes = []
    texts = text.replace('\r', '\n').split('\n')
    cur_texts = []
    (c_w, c_h), baseline = cv2.getTextSize('1234567890gj', font, font_scale, thickness)
    c_h = int(c_h + baseline)
    c_w = int(c_w / 12)
    for text in texts:
        cur_text = [text]
        while cur_text:
            text = cur_text.pop()
            if len(text) == 0:
                continue
            t_w, t_h = c_w * len(text), c_h
            if t_w > width and len(text) > 1:
                # 获取一个字符的
                mid = max(int(width / c_w), 1)
                cur_text.append(text[mid:])
                cur_text.append(text[:mid])
            else:
                x2 = max(x2, x1 + t_w)
                y2 += t_h + 2
                font_sizes.append([t_w, t_h])
                cur_texts.append(text)

    x1, _ = get_offset_coordinates(x1, x2, 0, width)
    y1, _ = get_offset_coordinates(y1, y2, 0, height)

    for text, (tw, th) in zip(cur_texts, font_sizes):
        left_x, right_x = x1, x1 + tw
        top_y, bottom_y = y1, y1 + th
        if bg_color != -1:
            try:
                cv2.rectangle(
                    img, (left_x - 1, top_y),
                    (right_x + 1, bottom_y),
                    bg_color, -1, cv2.LINE_AA
                )
            except:
                print("[put_text_using_opencv]: ", (left_x - 1, top_y), (right_x + 1, bottom_y))
        cv2.putText(
            img, text, (left_x, bottom_y - baseline),
            font, font_scale, text_color, thickness
        )
        x1, y1 = left_x, bottom_y + 1
    return img


def put_text_use_pillow(img, pts, text, tl, bg_color, text_color, zh_font_path):
    tl = max(tl, 10)
    if osp.exists(zh_font_path):
        font = ImageFont.truetype(zh_font_path, int(tl))
    else:
        zh_font_path = fm.findfont(fm.FontProperties(family="AR PL UKai CN"))
        if osp.exists(zh_font_path):
            font = ImageFont.truetype(zh_font_path, int(tl))
        else:
            print("[put_text_use_pillow]: 有中文, 但没有对应的字体.")
            font = None
    if font is None:
        return put_text_using_opencv(img, pts, text, tl, bg_color, text_color)

    height, width = img.shape[:2]
    is_gray = img.ndim == 2
    # ==========offset position=======
    x1, y1 = np.array(pts, dtype=int)
    x2, y2 = x1, y1
    font_sizes = []
    texts = text.replace('\r', '\n').split('\n')
    cur_texts = []
    for text in texts:
        cur_text = [text]
        while cur_text:
            text = cur_text.pop()
            if len(text) == 0:
                continue
            if pl_version < '9.5.0':  # 9.5.0 later
                left, top, right, bottom = font.getbbox(text)
                tw, th = right - left, bottom - top
            else:
                tw, th = font.getsize(text)
            if tw > width and len(text) > 1:
                mid = max(int(width / (tw / len(text))), 1)
                cur_text.append(text[mid:])
                cur_text.append(text[:mid])
            else:
                x2 = max(x2, x1 + tw)
                y2 += th + 2
                font_sizes.append([tw, th])
                cur_texts.append(text)
    x1, _ = get_offset_coordinates(x1, x2, 0, width)
    y1, _ = get_offset_coordinates(y1, y2, 0, height)
    if is_gray:
        img_pillow = Image.fromarray(img)
    else:
        img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pillow)
    for text, (tw, th) in zip(cur_texts, font_sizes):
        left_top_x, left_top_y = x1, y1
        right_bottom_x, right_bottom_y = x1 + tw, y1 + th
        if bg_color != -1:
            cv2.rectangle(
                img, (left_top_x, left_top_y - 1),
                (right_bottom_x, right_bottom_y + 1),
                bg_color, -1, cv2.LINE_AA
            )
        draw.text((left_top_x, left_top_y), text, font=font, fill=text_color)
        x1, y1 = left_top_x, right_bottom_y + 1
    img = np.asarray(img_pillow)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def norm_for_show(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255).astype(np.uint8)


def imshow(name, img, ori=False, t=0, cmp=113):
    """
    name: window name
    img: ndarray
    t: time step
    cmp: 113 is 'q', 27 is 'esc'
    """
    if img is not None:
        if not ori:
            height, width = img.shape[:2]
            if height > 2048 or width > 2048:
                img = size_pre_process(img, long=2048)
        cv2.imshow(name, img)
    key = cv2.waitKey(t)
    if key == cmp:
        exit()
    return key


def imwrite(path, img, overwrite=True):
    if not overwrite and osp.exists(path):
        print(f"{path} is existed!")
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# ============标签处处理==============


def create_labelme_file(png_path, content=None, overwrite=False, labelme_version="5.0.1"):
    json_path = osp.splitext(png_path)[0] + '.json'
    if osp.exists(json_path) and not overwrite:
        return

    # Create the base_info dictionary
    if content is None:
        content = create_labelme_content(
            None, png_path, [], labelme_version
        )

    # Write the base_info dictionary to a json file
    with open(json_path, 'w') as fo:
        json.dump(content, fo)


def create_labelme_content(img, png_path, shapes=[], labelme_version="5.0.1"):
    # # Convert the image to base64
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


def create_labelme_shape(label: str, pts, stype: str):
    pts = np.reshape(pts, [-1, 2]).squeeze().tolist()
    return {
        "label": label,
        "points": pts,
        "group_id": None,
        "shape_type": stype,
        "flags": {}
    }


def compute_polygon_from_mask(mask):
    """给一张mask图，输出其轮廓点，mask图必须是0-1"""
    import skimage

    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    if len(contours) == 0:
        print("No contour found, so returning empty polygon.")
        return []

    POLYGON_APPROX_TOLERANCE = 0.004
    ans = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygon = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
        )
        polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
        polygon = polygon[:-1]  # drop last point that is duplicate of first point
        if 0:
            vision = (255 * np.stack([mask] * 3, axis=-1)).astype(np.uint8)
            for y, x in polygon.astype(int):
                cv2.circle(vision, (x, y), 3, (0, 0, 222), -1)
            imshow("p", vision)
        ans.append(polygon[:, ::-1])  # yx -> xy
    return ans


class LabelObject(object):
    type = None
    pts = None
    ori_pts = None
    pts_normed = None
    label = None
    box = None
    height = None
    width = None

    def __str__(self):
        return f"type: {self.type}, label: {self.label}"


def parse_json(path, polygon, return_dict=False) -> [list, np.ndarray, str]:
    assert path.endswith('.json')
    info = json.load(open(path, "r"))
    base64_str = info.get("imageData", None)
    if base64_str is None:
        img = cv2.imread(path.replace(".json", ".png"))
    else:
        img_str = base64.b64decode(base64_str)
        np_arr = np.fromstring(img_str, np.uint8)
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


def show_yolo_label2(img, lines, xywh=True, classes: dict = None, colors=None, thickness=2):
    if classes is None:
        classes = {}
        for i in range(10):
            classes[i] = i
    if colors is None:
        colors = make_color_table(len(classes))
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    pts = []
    for line in lines:
        if not line: continue
        sp = line.strip().split(" ")
        idx, a, b, c, d = [float(x) for x in sp]
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
        img = put_text(img, classes[idx], (x1, y1), (0, 0, 0), (222, 222, 222))
        pts.append([idx, x1, y1, x2, y2])
    if thickness == -1:
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 1)
    return img, pts


def show_yolo_label(img, lines, xywh=True, classes: dict = None, colors=None, thickness=2):
    if classes is None:
        classes = {}
        for i in range(10):
            classes[i] = i
    if colors is None:
        colors = make_color_table(len(classes))
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    for line in lines:
        if not line: continue
        sp = line.strip().split(" ")
        idx, a, b, c, d = [float(x) for x in sp]
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
        img = put_text(img, classes[idx], (x1, y1), (0, 0, 0), (222, 222, 222))
    if thickness == -1:
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 1)
    return img


def show_yolo_file2(jpg_path, xywh=True, classes=None, colors=None, thickness=2):
    img = cv2.imread(jpg_path)
    txt = osp.splitext(jpg_path)[0] + ".txt"
    with open(txt, "r") as fo:
        lines = fo.readlines()
    img, pts = show_yolo_label2(img, lines, xywh, classes, colors, thickness)
    img = put_text(img, osp.basename(jpg_path))
    return img, pts


def show_yolo_file(jpg_path, xywh=True, classes=None, colors=None, thickness=2):
    img = cv2.imread(jpg_path)
    txt = osp.splitext(jpg_path)[0] + ".txt"
    with open(txt, "r") as fo:
        lines = fo.readlines()
    img, pts = show_yolo_label2(img, lines, xywh, classes, colors, thickness)
    img = put_text(img, osp.basename(jpg_path))
    return img


# =========Warp face from insightface=======
def estimate_norm(lmk, image_size=112):
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
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)  # [3,3]
    M = tform.params[0:2, :]  # 前两行
    return M


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def warp_face(img, x1, y1, x2, y2):
    cx, cy, fw, fh = xyxy2xywh([x1, y1, x2, y2]).flatten().astype(int)
    _scale = 256 / (max(fw, fh) * 1.5)
    return transform(img, (cx, cy), 256, _scale, 0)


def transform(data, center, output_size, scale, rotation):
    from skimage import transform as trans

    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def center_crop_rectangle(img, x1, y1, x2, y2, ratio=1.0):
    height, width = img.shape[:2]
    cx, cy, fw, fh = xyxy2xywh([x1, y1, x2, y2]).flatten().astype(int)
    hfs = max(fw, fh) // 2 * ratio
    bx, by, ex, ey = cx - hfs, cy - hfs, cx + hfs, cy + hfs
    bx, ex = get_offset_coordinates(bx, ex, 0, width)
    by, ey = get_offset_coordinates(by, ey, 0, height)
    bx, by, ex, ey = [int(x) for x in [bx, by, ex, ey]]
    return img[by:ey, bx:ex, :], bx, by


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


# =======3D==========
def pixel2d2camera3d2(pt_2d, z, mtx, dist):
    """
    pt_2d ： 畸变图像上的uv坐标
    """
    if np.ndim(z) > 1:
        x, y = map(lambda x: int(x), pt_2d)
        z = z[y, x]
    if z == 0:
        return None, None
    u_distorted, v_distorted = pt_2d
    homo_uv1 = np.array([u_distorted, v_distorted, 1])
    pc_xy1 = np.linalg.inv(mtx) @ homo_uv1  # p camera
    pc_xyz = pc_xy1 * z
    return pc_xyz, z


def warp_pts_with_homo(x, y, mtx):
    homo = np.array([x, y, 1]).T
    home = mtx @ homo
    home /= home[2]
    return home


def pixel2d2camera3d_numpy(p_uv, z, mtx):
    """

    :param p_uv:, Nx2
    :param z: N,
    :param mtx: 3x3
    :return: pc_xyz: (N, 3)
    """
    if np.ndim(z) == 2:
        idxs = np.reshape(p_uv, (-1, 2)).astype(int)
        z = z[idxs[:, 1], idxs[:, 0]]
    u_distorted, v_distorted = np.split(p_uv, 2, -1)

    homo_uv1 = np.stack(
        [u_distorted, v_distorted, np.ones_like(u_distorted)], axis=1
    )  # [N, 3, 1]
    pc_xy1 = np.matmul(np.linalg.inv(mtx), homo_uv1)[..., 0]  # [N, 3, 1]
    pc_xyz = pc_xy1 * z[:, None]  # z shape : (N, )
    return pc_xyz, z


def write_img_and_txt(split_name, img, text):
    if split_name[-4] == '.':
        split_name = split_name[:-4]
    if img is not None:
        os.makedirs(osp.dirname(split_name), exist_ok=True)
        imwrite(split_name + ".png", img)
    if text is not None:
        os.makedirs(osp.dirname(split_name), exist_ok=True)
        with open(split_name + ".txt", "w") as fo:
            fo.write(text)


def draw_gaze_with_k(
        image, start, pitchyaw, length, thickness=1, color=(0, 0, 255), is_degree=False
):
    """Draw gaze angle on given image with a given eye positions.

    pitchyaw: x朝右, y朝上
    pixel: x朝右, y朝下
    """
    if is_degree:
        pitchyaw = np.deg2rad(pitchyaw)
    pitch, yaw = pitchyaw
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
    k = np.rad2deg(np.arctan2(dy, dx))
    return image, k


def draw_gaze(
        image, start, pitchyaw, length, thickness=1, color=(0, 0, 255), is_degree=False
):
    image, _ = draw_gaze_with_k(
        image, start, pitchyaw, length, thickness, color, is_degree
    )
    return image


def gaze3dTo2d(gaze3d, M=None):
    if M is not None:
        gaze3d = np.dot(M, gaze3d)
    gaze3d = gaze3d / np.linalg.norm(gaze3d)
    dx, dy, dz = gaze3d
    pitch = np.rad2deg(np.arcsin(-dy))  # -dy: 向上为正
    yaw = np.rad2deg(np.arctan(-dx / (dz + 1e-7)))  # -dx 表示向左为正
    return pitch, yaw


def gaze2dTo3d(pitch, yaw, is_degree=True):
    """
    右手定则： x朝右, y朝上, z指向相机
    """
    if is_degree:
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    pitch = np.reshape(pitch, (-1, 1))
    yaw = np.reshape(yaw, (-1, 1))
    batch = np.shape(pitch)[0]
    gaze = np.zeros((batch, 3))
    gaze[:, 0] = np.cos(pitch) * np.sin(yaw)
    gaze[:, 1] = -np.sin(pitch)
    gaze[:, 2] = -np.cos(pitch) * np.cos(yaw)
    gaze = gaze / np.linalg.norm(gaze, axis=1, keepdims=True)
    return gaze


def cosine_similarity_deg(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    ab = np.sum(a * b, axis=1)
    ab = np.clip(ab, a_min=-float("inf"), a_max=0.999999)
    loss_rad = np.arccos(ab)  # rad
    return np.rad2deg(loss_rad)


def compute_euler(rotation_vector, translation_vector):
    """
    此函数用于从旋转向量计算欧拉角
    :param rotation_vector: 输入为旋转向量
    :return: 返回欧拉角在三个轴上的值
    """
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch = eulerAngles[0]
    yaw = eulerAngles[1]
    roll = eulerAngles[2]
    rot_params = np.array([pitch, yaw, roll]).flatten()
    return rot_params


class NormalWarp:
    def __init__(self, c_mtx, c_dist, distance_norm, focal_norm):
        self.c_mtx, self.c_dist = c_mtx, c_dist
        self.c_mtx_inv = np.linalg.inv(self.c_mtx)

        self.face_pts = np.array([
            [-45.0968, -21.3129, 21.3129, 45.0968, -26.2996, 26.2996],
            [-0.4838, 0.4838, 0.4838, -0.4838, 68.595, 68.595],
            [2.397, -2.397, -2.397, 2.397, -0.0, -0.0]
        ])
        self.face_pts_t = self.face_pts.T.reshape(-1, 1, 3)

        self.distance_norm = distance_norm  # normalized distance between eye and camera
        focal_norm = focal_norm  # focal length of normalized camera
        self.roiSize = (448, 448)  # size of cropped eye image
        self.n_ctx = np.array([
            [focal_norm, 0, self.roiSize[0] / 2],
            [0, focal_norm, self.roiSize[1] / 2],
            [0, 0, 1.0],
        ])

    def estimate_head_pose(self, landmarks, iterate=True):
        landmarks = np.reshape(landmarks, (-1, 2))
        ret, rvec, tvec = cv2.solvePnP(
            self.face_pts_t, landmarks, self.c_mtx, self.c_dist, flags=cv2.SOLVEPNP_EPNP
        )

        # further optimize
        if iterate:
            ret, rvec, tvec = cv2.solvePnP(
                self.face_pts_t, landmarks, self.c_mtx, self.c_dist, rvec, tvec, True
            )
        head_euler = compute_euler(rvec, tvec)
        return rvec, tvec, head_euler

    def __call__(self, image, landmarks):
        hr, ht, _ = self.estimate_head_pose(landmarks)

        # compute estimated 3D positions of the landmarks
        ht = np.repeat(ht, 6, axis=1)  # 6 points
        hR = cv2.Rodrigues(hr)[0]  # converts rotation vector to rotation matrix
        Fc = np.dot(hR, self.face_pts) + ht  # 3D positions of facial landmarks
        face_center = np.sum(Fc, axis=1, dtype=np.float32) / 6.0  # 人脸中心点

        # ---------- normalize image ----------
        distance = np.linalg.norm(face_center)  # actual distance face center and original camera

        # 计算新的坐标系，右眼指向左眼为x轴；嘴巴的中点向眼睛连线作垂线，向下为y轴；垂直于xy为z轴，方向是相机指向新坐标系原点
        face_center /= distance
        forward = face_center.reshape(3)
        hR = cv2.Rodrigues(hr)[0]
        hRx = hR[:, 0]
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        m_mtx = np.c_[right, down, forward].T  # rotation matrix R

        s_mtx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, self.distance_norm / distance],
        ])
        w_mtx = np.dot(
            np.dot(self.n_ctx, s_mtx),
            np.dot(m_mtx, self.c_mtx_inv)
        )  # (C_norm . M . C_c-1)

        # ---------裁剪人脸图片---------
        face_image = cv2.warpPerspective(image, w_mtx, self.roiSize)
        return face_image, m_mtx, w_mtx


# =====================deprecated==========
@deprecated
def create_labelme_json(img, basename, shapes):
    base64_str = cv2.imencode(".jpg", img)[1]
    height, width = img.shape[:2]
    return {
        "version": "5.2.0.post4",
        "flags": {},
        "shapes": shapes,
        "imagePath": basename,
        "imageData": base64.b64encode(base64_str).decode(),
        "imageHeight": height,
        "imageWidth": width,
    }


@deprecated
def move_txt_jpg(path, dst_folder, copy=True, do=False, postfixes=None):
    if postfixes is None:
        postfixes = [".txt", ".jpg", ".png"]
    os.makedirs(dst_folder, exist_ok=True)
    prefix = osp.splitext(path)[0]
    dirname = osp.dirname(prefix)
    basename = osp.basename(prefix)
    for postfix in postfixes:
        src = osp.join(dirname, basename + postfix)
        if osp.exists(src):
            dst = osp.join(dst_folder, basename + postfix)
            if not osp.exists(dst):
                if do:
                    if copy:
                        shutil.copy(src, dst)
                    else:
                        shutil.move(src, dst)
                else:
                    print("[move_txt_jpg]: ", src, dst)


@deprecated
def get_img_base64(img, quality=100):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    base64_str = cv2.imencode(".jpg", img, img_param)[1]
    return base64.b64encode(base64_str).decode()


@deprecated
def make_labelme_shape(label: str, pts, stype: str):
    pts = np.reshape(pts, [-1, 2]).squeeze().tolist()
    return {
        "label": label,
        "points": pts,
        "group_id": None,
        "shape_type": stype,
        "flags": {}
    }


@deprecated
def img2str(img):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    base64_str = cv2.imencode(".jpg", img, img_param)[1]
    return base64.b64encode(base64_str).decode()


@deprecated
def pixel2d2camera3d(pt_2d, z, mtx, dist):
    if np.ndim(z) > 1:
        x, y = map(lambda x: int(x), pt_2d)
        z = z[y, x]
    if z == 0:
        return None, None
    k1, k2, p1, p2, k3 = np.squeeze(dist)
    cx, cy = mtx[0, 2], mtx[1, 2]
    fx, fy = mtx[0, 0], mtx[1, 1]
    u, v = pt_2d
    x, y = (u - cx) / fx, (v - cy) / fy

    # ===============================
    r = np.sqrt(x ** 2 + y ** 2)
    k = 1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6
    xy = x * y
    x_distorted = x * k + 2 * p1 * xy + p2 * (r ** 2 + 2 * x ** 2)
    y_distorted = y * k + 2 * p2 * xy + p1 * (r ** 2 + 2 * y ** 2)
    # =====x -> x_distorted=========y -> y_distorted=================

    u_distorted = fx * x_distorted + cx
    v_distorted = fy * y_distorted + cy

    homo_uv1 = np.array([u_distorted, v_distorted, 1])
    pc_xy1 = np.linalg.inv(mtx) @ homo_uv1  # p camera
    pc_xyz = pc_xy1 * z
    return pc_xyz, z


@deprecated
def image_norm(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-3)
    return (image * 255).astype(np.uint8)


@deprecated
def parse_json_dict(path):
    try:
        fo = open(path, "r")
        info = json.load(fo)
        fo.close()
    except Exception as why:
        print("[parse_json] \nwhy?\n", why)
        print(f"error: {path}")
        return [], None, ""
    base64_str = info.get("imageData", None)
    all_info = defaultdict(list)
    if base64_str is None:
        img = None
    else:
        img_str = base64.b64decode(base64_str)
        np_arr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        image_height = info.get("imageHeight", None)
        image_width = info.get("imageWidth", None)
    else:
        image_height, image_width = img.shape[:2]
    for shape in info.get("shapes", []):
        obj = LabelObject()
        obj.label = shape.get("label", None)
        obj.pts = shape.get("points", [])
        obj.type = shape.get("shape_type", "")
        obj.height = image_height
        obj.width = image_width
        # =====processed=======
        obj.pts_normed = np.reshape(obj.pts, [-1, 2]) / np.array(
            [image_width, image_height]
        )
        all_info[obj.label].append(obj)
    return all_info, img, osp.basename(path).split(".")[0]


@deprecated
def show_yolo_txt(jpg_path, xywh=True, classes=None, colors=None, thickness=2):
    img = cv2.imread(jpg_path)
    txt = osp.splitext(jpg_path)[0] + ".txt"
    with open(txt, "r") as fo:
        lines = fo.readlines()
    img = show_yolo_label(img, lines, xywh, classes, colors, thickness)
    img = put_text(img, osp.basename(jpg_path))
    return img

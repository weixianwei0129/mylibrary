import glob
import os
import cv2
import time
import json
import random
import base64
import shutil
import hashlib
import requests
import numpy as np
from io import BytesIO
from functools import partial
import matplotlib.pylab as plt
from multiprocessing import Pool

from PIL import Image
from PIL import ImageFont, ImageDraw
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.font_manager as fm
from functools import partial

np.random.seed(123456)
random.seed(123456)


class ONNXRunner:
    def __init__(self, path):
        import onnxruntime

        providers = [
            "CoreMLExecutionProvider"
        ]  # 'CUDAExecutionProvider' 'CPUExecutionProvider'
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        print("inputs: ", [x.name for x in self.session.get_inputs()])
        print("outputs: ", [x.name for x in self.session.get_outputs()])

    def run(self, img):
        try:
            return self.session.run(
                [x.name for x in self.session.get_outputs()],
                {self.session.get_inputs()[0].name: img},
            )
        except Exception as e:
            print("ONNXRunner why?: ")
            print(e)
            print(self.session.get_inputs()[0])
            print(img.shape)


def calculate_md5(file_path):
    with open(file_path, "rb") as file:
        md5_hash = hashlib.md5()
        while True:
            data = file.read(4096)  # 每次读取4KB数据
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()


def read_avif_img(path):
    AVIFimg = Image.open(path)
    img = np.array(AVIFimg)
    return img[..., ::-1]


def img2str(img):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    base64_str = cv2.imencode(".jpg", img, img_param)[1]
    return base64.b64encode(base64_str).decode()


def pad_image(img, target=None, value=(0, 0, 0), centre=True):
    height, width = img.shape[:2]
    long_side = max(height, width)
    if target is None:
        t_h = t_w = long_side
    else:
        t_w, t_h = target
    top, left = 0, 0
    if centre:
        top = (t_h - height) // 2
        left = (t_w - width) // 2
    bottom, right = t_h - height - top, t_w - width - left
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
    )
    return img, left, top


def divisibility(a, r=32):
    if r == 1:
        return int(a)
    return int(np.ceil(a / r) * r)


def size_pre_process(img, longest=4096, **kwargs):
    """kwargs
    interpolation

    """
    align_fun = partial(divisibility, r=kwargs.get('align', 32))
    h, w = img.shape[:2]
    if "hard" in kwargs:
        rw = rh = align_fun(kwargs["hard"])
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
        print(f"rw({rw}->{longest})")
        rw = longest
    if rh > longest:
        print(f"rh({rh}->{longest})")
        rh = logging
    interpolation = kwargs.get("interpolation", None)
    if interpolation is None:
        if rw * rh > h * w:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
    return cv2.resize(img, (rw, rh), interpolation=interpolation)


def get_img_base64(img, quality=100):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    base64_str = cv2.imencode(".jpg", img, img_param)[1]
    return base64.b64encode(base64_str).decode()


def paint_chinese_opencv(im, text, tl, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("Songti.ttc", tl)
    size = font.getsize(text)

    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img, size


def get_offset_coordinates(v1, v2, v_min, v_max):
    v2_add = max(0, v_min - v1)
    v1 = max(v_min, v1)
    v1_sub = max(0, v2 - v_max)
    v2 = min(v_max, v2)
    v1 -= v1_sub
    v2 += v2_add
    return v1, v2


def imshow(name, img, t=0, cmp=113):
    """
    name: window name
    img: ndarray
    t: time step
    cmp: 113 is 'q', 27 is 'esc'
    """
    if img is not None:
        cv2.imshow(name, img)
    key = cv2.waitKey(t)
    if key == cmp:
        exit()
    return key


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def is_chinese(string):
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def put_text(
        im0,
        text,
        pts,
        bg_color=None,
        text_color=None,
        tl=None,
        chinese_font="resource/Songti.ttc",
        alpha=1.0
):
    """

    alpha: 不透明度
    """
    # ============config========
    if is_chinese(text):
        if not os.path.exists(chinese_font):
            print(f"有中文, 但没有对应的字体 'resource/Songti.ttc'. ")
        else:
            font = ImageFont.truetype(chinese_font, tl)

    if bg_color is None:
        bg_color = (0, 0, 0)
    if text_color is None:
        text_color = (255, 255, 255)
    height, width = im0.shape[:2]
    if tl is None:
        tl = round(0.02 * np.sqrt(height ** 2 + width ** 2)) + 1
    en_path = fm.findfont(fm.FontProperties(family="Arial"))
    font = ImageFont.truetype(en_path, tl)

    # ==========offset position=======
    x1, y1 = np.array(pts, dtype=int)
    x2, y2 = x1, y1
    font_sizes = []
    texts = text.replace('\r', '\n').split('\n')
    for text in texts:
        tw, th = font.getsize(text)
        font_sizes.append([tw, th])
        x2 = max(x2, x1 + tw)
        y2 += th + 2

    x1, _ = get_offset_coordinates(x1, x2, 0, width)
    y1, _ = get_offset_coordinates(y1, y2, 0, height)
    img = im0.copy()
    for text, (tw, th) in zip(texts, font_sizes):
        left_top_x, left_top_y = x1, y1
        right_bottom_x, right_bottom_y = x1 + tw, y1 + th
        if bg_color != -1:
            cv2.rectangle(
                img, (left_top_x, left_top_y - 1),
                (right_bottom_x, right_bottom_y + 1),
                bg_color, -1, cv2.LINE_AA
            )
        img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pillow)
        draw.text((left_top_x, left_top_y), text, font=font, fill=text_color)
        img = cv2.cvtColor(np.asarray(img_pillow), cv2.COLOR_RGB2BGR)
        x1, y1 = left_top_x, right_bottom_y + 1
    im0 = cv2.addWeighted(im0, (1 - alpha), img, alpha, 0)
    return im0


def download_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return img


def make_img_smaller(img, height):
    if height is not None and img.shape[0] > height:
        img = size_pre_process(img, height=height)
    return img


def multi_download(all_info, num_thread, func, **kwargs):
    """

    Args:
        all_info:
        num_thread:
        **kwargs:

    Returns:

    """
    if num_thread == 1:
        func([0, all_info], **kwargs)
    else:
        begin = 0
        total = len(all_info)
        interval = int(np.ceil(total / num_thread))
        end = interval

        in_args = []
        index = 0
        while begin < total:
            in_args.append([index, all_info[begin:end]])
            begin += interval
            end += interval
            index += 1
        pool = Pool(num_thread)
        pool.map(partial(func, **kwargs), in_args)


def timer(func):
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


def move_txt_jpg(path, dst_folder, copy=True, do=False):
    os.makedirs(dst_folder, exist_ok=True)
    prefix = os.path.splitext(path)[0]
    dirname = os.path.dirname(prefix)
    basename = os.path.basename(prefix)
    for postfix in [".txt", ".jpg"]:
        src = os.path.join(dirname, basename + postfix)
        dst = os.path.join(dst_folder, basename + postfix)
        if do:
            if copy:
                shutil.copy(src, dst)
            else:
                shutil.move(src, dst)
        else:
            print(src, dst)


def delete_txt_jpg(path, do=False):
    prefix = os.path.splitext(path)[0]
    dirname = os.path.dirname(prefix)
    basename = os.path.basename(prefix)
    for postfix in [".txt", ".jpg"]:
        src = os.path.join(dirname, basename + postfix)
        if do:
            os.remove(src)
        else:
            print("delete: ", src)


class LabelObject(object):
    type = None
    pts = None
    ori_pts = None
    pts_normed = None
    label = None
    box = None
    height = None
    width = None


def parse_json(path, polygon) -> [list, np.ndarray, str]:
    info = json.load(open(path, "r"))
    base64_str = info.get("imageData", None)
    if base64_str is None:
        img = cv2.imread(path.replace(".json", ".jpg"))
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
    basename = os.path.basename(path).split(".")[0]
    return obj_list, img, basename


def parse_json_dict(path):
    try:
        fo = open(path, "r")
        info = json.load(fo)
        fo.close()
    except Exception as why:
        print("\nwhy?\n", why)
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
    return all_info, img, os.path.basename(path).split(".")[0]


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def warp_regions(img, box):
    p0, p1, p2, p3 = box
    h = int(distance(p0, p3))
    w = int(distance(p0, p1))
    pts1 = np.float32([p0, p1, p3])
    pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    return cv2.warpAffine(img, cv2.getAffineTransform(pts1, pts2), (w, h))


def random_color(amin, amax):
    b = np.random.randint(amin, amax)
    g = np.random.randint(amin, amax)
    r = np.random.randint(amin, amax)
    return b, g, r


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


def rotate_image(img, angle, point=None, scale=1.0):
    """逆时针旋转为正"""
    height, width = img.shape[:2]
    if point is None:
        point = (width // 2, height // 2)
    rotate_mtx = cv2.getRotationMatrix2D(point, angle, scale)
    return cv2.warpAffine(img, rotate_mtx, (width, height), borderMode=cv2.BORDER_REPLICATE)


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


def make_labelme_json(img, basename, shapes):
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


def show_yolo_txt(path, classes=None, colors=None, thickness=2):
    if classes is None:
        classes = {}
        for i in range(10):
            classes[i] = i
    if colors is None:
        colors = make_color_table(len(classes))
    img = cv2.imread(path)
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    txt = os.path.splitext(path)[0] + ".txt"
    with open(txt, "r") as fo:
        lines = fo.readlines()
    for line in lines:
        sp = line.strip().split(" ")
        idx, cx, cy, w, h = [float(x) for x in sp]
        x1, y1, x2, y2 = (
                xywh2xyxy([cx, cy, w, h]) * np.array([width, height, width, height])
        ).astype(int)
        if thickness == -1:
            mask = cv2.rectangle(mask, (x1, y1), (x2, y2), colors[idx], thickness)
        else:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[idx], thickness)
        img = put_text(img, classes[idx], (x1, y1), (0, 0, 0), (222, 222, 222))
    img = cv2.addWeighted(img, 0.7, mask, 0.3, 1)
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
    return warped


def norm_crop2(img, landmark, image_size=112):
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


def crop_rectangle(img, x1, y1, x2, y2):
    height, width = img.shape[:2]
    cx, cy, fw, fh = xyxy2xywh([x1, y1, x2, y2]).flatten().astype(int)
    hfs = max(fw, fh) // 2
    bx, by, ex, ey = cx - hfs, cy - hfs, cx + hfs, cy + hfs
    bx, ex = get_offset_coordinates(bx, ex, 0, width)
    by, ey = get_offset_coordinates(by, ey, 0, height)
    return img[by:ey, bx:ex, :], bx, by


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


# =======camera==========
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


def warp_pts_with_homo(x, y, mtx):
    homo = np.array([x, y, 1]).T
    home = mtx @ homo
    home /= home[2]
    return home


def pixel2d2camera3d_numpy(p_uv, z, mtx, dist):
    """

    :param p_uv:, Nx2
    :param z: N,
    :param mtx: 3x3
    :param dist: 5x1
    :return: pc_xyz: (N, 3)
    """
    if np.ndim(z) == 2:
        idxs = np.reshape(p_uv, (-1, 2)).astype(int)
        z = z[idxs[:, 1], idxs[:, 0]]
    k1, k2, p1, p2, k3 = np.squeeze(dist)
    cx, cy = mtx[0, 2], mtx[1, 2]
    fx, fy = mtx[0, 0], mtx[1, 1]
    u, v = np.split(p_uv, 2, -1)
    x, y = (u - cx) / fx, (v - cy) / fy

    r = np.sqrt(x ** 2 + y ** 2)
    k = 1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6
    xy = x * y
    x_distorted = x * k + 2 * p1 * xy + p2 * (r ** 2 + 2 * x ** 2)
    y_distorted = y * k + 2 * p2 * xy + p1 * (r ** 2 + 2 * y ** 2)

    u_distorted = fx * x_distorted + cx
    v_distorted = fy * y_distorted + cy

    homo_uv1 = np.stack(
        [u_distorted, v_distorted, np.ones_like(u_distorted)], axis=1
    )  # [N, 3, 1]
    pc_xy1 = np.matmul(np.linalg.inv(mtx), homo_uv1)[..., 0]  # [N, 3, 1]
    pc_xyz = pc_xy1 * z[:, None]  # z shape : (N, )
    return pc_xyz, z


def calibrate_single_camera(pattern, height, width, cols, rows, wk=-1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    if isinstance(pattern, str):
        all_path = glob.glob(pattern)
        all_path.sort()
    elif isinstance(pattern, list):
        all_path = pattern
    else:
        raise TypeError(pattern)
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in left image plane.
    total = 0
    for path in all_path:
        img = cv2.imread(path)[:height, :width, :]

        # Using OpenCV
        # Find the chess board corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            total += 1
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_p)
            img_points.append(corners)

            if wk >= 0:
                print(path)
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        if wk >= 0:
            cv2.imshow("draw", img)
            cv2.waitKey(wk)
    print("Using...", total)
    ret, mtx, dist, r_vecs, t_vecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None
    )
    print("CameraCalibrate:")
    print("MRS: ", ret)
    print(mtx)
    print(dist)
    print("=" * 40)
    return ret, mtx, dist


def write_img_and_txt(split_name, img, text):
    if split_name[-4] == '.':
        split_name = split_name[:-4]
    if img is not None:
        os.makedirs(os.path.dirname(split_name), exist_ok=True)
        imwrite(split_name + ".png", img)
    if text is not None:
        with open(split_name + ".txt", "w") as fo:
            os.makedirs(os.path.dirname(split_name), exist_ok=True)
            fo.write(text)


def draw_gaze(
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


# =================Train===============


class LogHistory:
    def __init__(self, log_dir):
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


# =============Train===========


def set_file_only_read(filename):
    # 获取当前文件的权限
    current_permissions = os.stat(filename).st_mode

    # 设置文件为只读
    os.chmod(filename, current_permissions & ~0o222)  # 将写权限（w）移除

    # 检查文件权限
    new_permissions = os.stat(filename).st_mode
    if not new_permissions & 0o222:
        print(f"{filename}设置为只读成功！")
    else:
        print(f"{filename}设置为只读失败！")


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


def center_crop(bbox, ratio, width, height):
    (x1, y1, x2, y2) = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    fw, fh = abs(x2 - x1), abs(y2 - y1)
    side = max(fw, fh) * ratio
    x1, x2 = get_offset_coordinates(
        cx - side / 2, cx + side / 2, 0, width - 1
    )
    y1, y2 = get_offset_coordinates(
        cy - side / 2, cy + side / 2, 0, height - 1
    )
    x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
    return x1, y1, x2, y2


def GetEuler(rotation_vector, translation_vector):
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
    def __init__(self, c_mtx, c_dist, distance_norm=1000, focal_norm=1600):
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
        head_euler = GetEuler(rvec, tvec)
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

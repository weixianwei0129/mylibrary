import os

import cv2
try:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
except:
    pass
from PyQt5.QtGui import QImage, QPixmap, QDesktopServices
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QMessageBox


def pop_info_box(info: str, title: str) -> None:
    """
    弹出一个信息框。

    :param info: 信息框中的文本内容
    :param title: 信息框的标题
    """
    msg_box = QMessageBox()
    msg_box.setText(info)
    msg_box.setWindowTitle(title)
    msg_box.setIcon(QMessageBox.Information)
    msg_box.addButton(QMessageBox.Ok)
    msg_box.exec_()


def get_scaled_size(original_width: int, original_height: int, max_width: int, max_height: int) -> (int, int):
    """
    计算图像的缩放尺寸。

    :param original_width: 图像的原始宽度
    :param original_height: 图像的原始高度
    :param max_width: 图像的最大宽度
    :param max_height: 图像的最大高度
    :return: 新的宽度和高度
    """
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    ratio = min(width_ratio, height_ratio)

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    return new_width, new_height


def cv2pix(frame, width: int, height: int) -> QPixmap:
    """
    将一个OpenCV图像帧转换为一个Qt的QPixmap对象。

    :param frame: 输入的OpenCV图像帧
    :param width: 输出的QPixmap对象的宽度
    :param height: 输出的QPixmap对象的高度
    :return: 缩放后的QPixmap对象
    """
    if frame is None:
        raise ValueError("Frame cannot be None.")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], rgb_frame.shape[1] * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    scaled_pixmap = QPixmap(pixmap).scaled(width, height)

    return scaled_pixmap

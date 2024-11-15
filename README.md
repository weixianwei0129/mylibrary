# MyLibrary

- 一些常用的基于OpenCV以及PyTorch的脚本库, 不定期更新.

## Usage

- clone this repo and `cd mylibrary`
- `pip install -e .`

## Introduce

- `common.py`: 大部分功能在此文件中, 主要包括图像显示, 常用预处理, 视频流控制等.
    - 用法: `import wxw.common as cm`.
    - 常用功能有:
        - `size_pre_process`: 用于图片多种方式的resize;
        - `put_text`: 用于图片上写log, 支持字符串`\n`换行显示;
        - `imshow` & `imwrite`: 类似于OpenCV, 但可以显示列表图片,可自动排列;
            - 搭配`ControlKey`可以自动控制视频流快进,后退等;
        - `LabelObject` & `parse_json`: 用于解析**LabelMe**软件的图片标签;
- `all_in_one.py`: 主要包含常用头文件, 用于写**临时脚本**时导入.
    - 用法: `from wxw.all_in_one import *`.
    - 注意: 工程代码禁用.
- `model_utils.py`: 模型训练时一些自定义功能,Writer & LR 等.
- `qt5.py`: `qt5`相关功能.
- `scripts/*`: 脚本文件夹,包括从百度图库爬取图片,相机内参标定,人物换脸等脚本.

## Release

```commandline
python setup.py sdist
twine upload dist/wxw-x.x.x.tar.gz --verbose
```
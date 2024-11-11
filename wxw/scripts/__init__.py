import os
import glob
import os.path as osp

import cv2
from tqdm import tqdm
import wxw.common as cm
from wxw.scripts.cameras import (
    collect, calibrate_single_camera,
    video2images, multi_video2images,
)


def pack_files(pattern, files_per_folder=500, being=0, sort_method=None, **kwargs):
    """
    Pack files into folders based on a specified pattern and interval.

    Args:
        pattern (str): The glob pattern to match files.
        files_per_folder (int, optional): The number of files per folder. Defaults to 500.
        being (int, optional): The starting index for folder naming. Defaults to 0.
        sort_method (callable, optional): A function to sort the files. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the move_file_pair function.

    Returns:
        None
    """
    all_path = glob.glob(pattern, recursive=True)
    if sort_method:
        all_path.sort(key=sort_method)
    else:
        all_path.sort()
    print("total: ", len(all_path))

    for i, p in tqdm(enumerate(all_path)):
        new_folder = osp.join(osp.dirname(p), f"{i // files_per_folder + being}")
        cm.move_file_pair(p, new_folder, copy=False, **kwargs)

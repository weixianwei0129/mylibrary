import os
import cv2
import glob
import os.path as osp
from tqdm import tqdm
import wxw.common as cm


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
        folder = osp.splitext(video)[0]
        saved = 0

        if osp.exists(folder):
            saved = len(glob.glob(osp.join(folder, "*.png")))
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
                new_path = osp.join(folder, f"{str(index).zfill(5)}.png")
                cv2.imwrite(new_path, frame)

            index += 1
            ret, frame = cap.read()

        cap.release()

    print(f"{thread_idx} Finished!")

import os
import cv2
import glob


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

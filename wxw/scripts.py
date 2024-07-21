import os
import cv2
import glob


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

import os
import os.path as osp

import cv2
import glob
import numpy as np
from tqdm import tqdm
import wxw.common as cm
from multiprocessing.pool import ThreadPool


def collect(path=0, flip=False, output_dir='process'):
    """Capture video frames and save them as images.

    This function captures video frames from the default camera, displays them,
    and saves them as images when specific keys are pressed.

    - Press 's' or space to save the current frame.
    - Press 'q' to quit the capture.

    The saved images are stored in the 'process' directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        if flip:
            frame = frame[:, ::-1, :]

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Create a copy of the frame for display
        display_frame = frame.copy()

        # Add frame dimensions text to the display frame
        info_text = f"{height}x{width}@saved{frame_index}"
        display_frame = cm.put_text(display_frame, info_text, (0, 0), background_color=-1)

        # Show the frame
        cv2.imshow("Frame", display_frame)

        # Handle key events
        key = cv2.waitKey(1)
        if key in [ord("s"), ord(" ")]:
            # Save the current frame
            image_path = osp.join(output_dir, f"{frame_index}.png")
            cm.imwrite(image_path, frame)
            frame_index += 1
            print(f"Saved frame {frame_index}")
        elif key == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def calibrate_single_camera(pattern, height, width, cols, rows, wait_key=-1):
    """Calibrate a single camera using chessboard images.

    Args:
        pattern (str or list): File pattern or list of image paths.
        height (int): Height of the images.
        width (int): Width of the images.
        cols (int): Number of columns in the chessboard.
        rows (int): Number of rows in the chessboard.
        wait_key (int, optional): Delay in milliseconds for displaying images. Default is -1.

    Returns:
        tuple: Calibration parameters including ret, mtx, dist, rvecs, and tvecs.
    """
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (cols-1,rows-1,0)
    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Get all image paths
    if isinstance(pattern, str):
        all_paths = glob.glob(pattern)
        all_paths.sort()
    elif isinstance(pattern, list):
        all_paths = pattern
    else:
        raise TypeError("Pattern must be a string or a list of strings.")

    # Arrays to store object points and image points from all the images
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane

    total_images = 0

    for path in all_paths:
        img = cv2.imread(path)[:height, :width, :]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            total_images += 1
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_p)
            img_points.append(corners)

            if wait_key >= 0:
                print(path)
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)

        if wait_key >= 0:
            cv2.imshow('Chessboard', img)
            cv2.waitKey(wait_key)

    print(f"Using {total_images} images for calibration.")

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height),
        None, None
    )

    print("Camera calibration results:")
    print(f"Reprojection error: {ret}")
    print(f"Camera matrix: {mtx.tolist()}")
    print(f"Distortion coefficients: {dist.tolist()}")
    print('=' * 40)

    return ret, mtx, dist, rvecs, tvecs


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


def multi_video2images(all_videos, interval=1):
    total = len(all_videos)
    print("total: ", total)

    def do(idx):
        video = all_videos[idx]
        folder = osp.splitext(video)[0]

        saved = 0
        if osp.exists(folder):
            saved = len(glob.glob(osp.join(folder, "*.png")))

        cap = cv2.VideoCapture(video)

        index = 0
        ret, frame = cap.read()
        while ret:
            index += 1
            if index > saved and index % interval == 0:
                new_path = osp.join(folder, f"{str(index).zfill(5)}.png")
                cm.imwrite(new_path, frame)
            ret, frame = cap.read()
        cap.release()

    with ThreadPool(cm.NUM_THREADS) as pool:
        with tqdm(pool.imap_unordered(do, range(total)), total=total) as pbar:
            for _ in pbar:
                pass


if __name__ == '__main__':
    calibrate_single_camera("process/*.png", 1080, 1920, 9, 6, 100)

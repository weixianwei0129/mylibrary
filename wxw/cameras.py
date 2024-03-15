import os
import cv2
import glob
import numpy as np
import wxw.common as cm


def collect():
    root = 'process'

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    index = 0
    while ret:
        frame = frame[:, ::-1, :]
        height, width = frame.shape[:2]
        show = frame.copy()
        info = f"{height}x{width}"
        show = cm.put_text(show, info, (0, 0), bg_color=-1)
        cv2.imshow("frame", show)
        key = cv2.waitKey(1)
        if key in [ord("s"), ord(" ")]:
            cm.imwrite(os.path.join(root, f"{index}.png"), frame)
            index += 1
            print("save ", index)
        elif key == ord('q'):
            break
        ret, frame = cap.read()


def calibrate_single_camera(pattern, height, width, cols, rows, wk=-1):
    # cols, rows = 11, 8
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
            cv2.imshow('draw', img)
            cv2.waitKey(wk)
    print("Using...", total)
    ret, mtx, dist, r_vecs, t_vecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)
    print("left camera info:")
    print("MRS: ", ret)
    print(mtx.tolist())
    print(dist.tolist())
    print('=' * 40)
    return ret, mtx, dist


if __name__ == '__main__':
    # collect()
    calibrate_single_camera("process/*.png", 1080, 1920, 9, 6, 100)

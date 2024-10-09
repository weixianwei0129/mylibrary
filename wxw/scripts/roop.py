import os
import glob

import cv2
import insightface
import numpy as np
from numpy.linalg import norm as l2norm

import wxw.common as cm


# np.random.seed(123456)


class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        # for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


def read_from_json(js_path):
    """Read face data from a JSON file and return bounding boxes and keypoints.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        tuple: A tuple containing bounding boxes and keypoints.
    """
    info, img, basename = cm.parse_json(js_path, return_dict=True)
    # FIXME: 目前只能有一个脸
    bboxes = np.array(info['face'][0].pts).flatten().tolist()
    bboxes = np.reshape(bboxes + [1.0], (1, 5))
    kpss = [info[str(x)][0].pts for x in range(1, 6)]
    kpss = np.reshape(kpss, (1, 5, 2))
    return bboxes, kpss


def analyze_face(fa, img, js_path=''):
    """Analyze faces in an image using a face analyzer model.

    Args:
        fa: The face analyzer model.
        img (numpy.ndarray): The input image.
        js_path (str, optional): Path to a JSON file for fallback detection. Defaults to ''.

    Returns:
        list: A list of detected faces with their attributes.
    """
    bboxes, kpss = fa.det_model.detect(
        img,
        max_num=0,
        metric='default'
    )
    if bboxes.shape[0] == 0 and os.path.isfile(js_path):
        bboxes, kpss = read_from_json(js_path)

    if bboxes.shape[0] == 0:
        return []
    # else: # debug
    #     print(bboxes.shape, kpss, kpss.shape)
    #     print(img.shape)
    #     for x1, y1, x2, y2, s in bboxes.astype(int):
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 222), 2)
    #     for i, (x, y) in enumerate(np.reshape(kpss, (-1, 2)).astype(int)):
    #         cv2.circle(img, (x, y), 2, (0, 0, 222), -1)
    #         img = cm.put_text(img, str(i), (x, y))
    #     cm.imshow("img", img)

    ret = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        for taskname, model in fa.models.items():
            if taskname in ['detection', 'landmark_2d_106', 'genderage']:
                continue
            model.get(img, face)
        ret.append(face)
    return ret


def got_src():
    """Select a random image, analyze the face, and return the first detected face.

    This function selects a random image from the current directory, analyzes the face,
    and returns the first detected face.

    Returns:
        dict: The first detected face's analysis result.
    """
    # Define the pattern to match image files
    pattern_a = "*.png"
    all_a = glob.glob(pattern_a)

    while True:
        # Select a random image path
        path_a = np.random.choice(all_a)
        source_image = cv2.imread(path_a)

        # Analyze the face in the image
        source_face = analyze_face(face_analyser, source_image)

        if len(source_face) > 0:
            return source_face[0]


if __name__ == '__main__':
    # 把图片A上的人脸,换到图片B上,最终生成图片C;图片C和图片B的只有脸的样子不一样,其他都一样.
    providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    model_path = "inswapper_128.onnx"
    face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    pattern_b = "*.jpg"
    all_path = glob.glob(pattern_b)
    for path_b in all_path:
        # 图片B
        target_img = cv2.imread(path_b)
        post_fix = os.path.splitext(path_b)[-1]
        json_b = path_b.replace(post_fix, '.json')
        target_faces = analyze_face(face_analyser, target_img, json_b)
        if len(target_faces) == 0:
            print(target_faces)
            print("path_b: ", path_b)
            cv2.imshow("target_img", target_img)
            cv2.waitKey(0)
        result = target_img
        for target_face in target_faces:
            # 图片C模板,从B上拷贝出来的
            result = face_swapper.get(result, target_face, got_src())
        cv2.imshow("img", result)
        cv2.waitKey(0)

import gdown
import skimage
import numpy as np
import onnxruntime


class EfficientSam:
    def __init__(self, encoder_path, decoder_path):
        self._encoder_session = onnxruntime.InferenceSession(encoder_path)
        self._decoder_session = onnxruntime.InferenceSession(decoder_path)

    def __call__(self, image, points: [[float, float]], points_labels: [int]):
        batched_images = (image[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        (_image_embedding,) = self._encoder_session.run(
            output_names=None,
            input_feed={"batched_images": batched_images},
        )
        mask = _compute_mask_from_points(self._decoder_session,
                                         image, _image_embedding,
                                         points, points_labels
                                         )
        return mask


def _compute_mask_from_points(
        decoder_session, image, image_embedding, points, point_labels
):
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array(point_labels, dtype=np.float32)

    # batch_size, num_queries, num_points, 2
    batched_point_coords = input_point[None, None, :, :]
    # batch_size, num_queries, num_points
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int64),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask = masks[0, 0, 0, :, :]  # (1, 1, 3, H, W) -> (H, W)
    mask = mask > 0.0

    MIN_SIZE_RATIO = 0.05
    skimage.morphology.remove_small_objects(
        mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
    )

    # if 1:
    #     imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
    return mask


def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


def compute_polygon_from_mask(mask):
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=_get_contour_length)
    POLYGON_APPROX_TOLERANCE = 0.004
    polygon = skimage.measure.approximate_polygon(
        coords=contour,
        tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
    )
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # drop last point that is duplicate of first point

    return polygon[:, ::-1]  # yx -> xy


class EfficientSamVitT(EfficientSam):
    name = "EfficientSam (speed)"

    def __init__(self):
        super().__init__(
            encoder_path=gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx",
                # NOQA
                md5="2d4a1303ff0e19fe4a8b8ede69c2f5c7",
            ),
            decoder_path=gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx",
                # NOQA
                md5="be3575ca4ed9b35821ac30991ab01843",
            ),
        )


class EfficientSamVitS(EfficientSam):
    name = "EfficientSam (accuracy)"

    def __init__(self):
        super().__init__(
            encoder_path=gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",
                # NOQA
                md5="7d97d23e8e0847d4475ca7c9f80da96d",
            ),
            decoder_path=gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx",
                # NOQA
                md5="d9372f4a7bbb1a01d236b0508300b994",
            ),
        )


if __name__ == '__main__':
    import cv2

    img = cv2.imread("/home/weixianwei/Desktop/IR_0000001_8.png")
    sam = EfficientSamVitS()
    points = [[188.91354466858792, 313.04899135446686], [272.7752161383285, 317.5158501440922],
              [186.17579250720462, 176.7377521613833], [128.97118155619597, 184.51873198847264],
              [252.31412103746396, 179.90778097982712], [300.4409221902017, 176.59365994236313],
              [260.8155619596542, 213.62536023054759]]
    points_labels = [1, 1, 1, 1, 1, 1, 1]
    mask = sam(img, points, points_labels)
    print(mask, polygon)
    cv2.imshow("mask", mask.astype(np.float32))
    cv2.waitKey(0)

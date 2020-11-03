import cv2
import numpy as np

from .model import Model
from .utils import Detection, resize_image, resize_image_letterbox



class YOLO(Model):
    class Params:
        # Magic numbers are copied from yolo samples
        def __init__(self, param, side):
            self.num = 3 if 'num' not in param else int(param['num'])
            self.coords = 4 if 'coords' not in param else int(param['coords'])
            self.classes = 80 if 'classes' not in param else int(param['classes'])
            self.side = side
            self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
                            30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
                            116.0, 90.0, 156.0, 198.0, 373.0, 326.0] \
                if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

            self.isYoloV3 = False

            if param.get('mask'):
                mask = [int(idx) for idx in param['mask'].split(',')]
                self.num = len(mask)

                masked_anchors = []
                for idx in mask:
                    masked_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = masked_anchors

                self.isYoloV3 = True  # Weak way to determine but the only one.

    def __init__(self, *args, keep_aspect_ratio=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.threshold = 0.5
        self.iou_threshold = 0.5
        self.keep_aspect_ratio = keep_aspect_ratio
        self.resize = resize_image_letterbox if self.keep_aspect_ratio else resize_image

        assert len(self.net.input_info) == 1, "Sample supports only YOLO V* based single input topologies"
        self.image_blob_name = next(iter(self.net.input_info))
        if self.net.input_info[self.image_blob_name].input_data.shape[1] == 3:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = True
        else:
            self.n, self.h, self.w, self.c = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = False
        assert self.n == 1, 'Only batch size == 1 is supported.'

        self.yolo_layer_params = {}
        for layer_name, out_blob in self.net.outputs.items():
            shape = out_blob.shape if len(out_blob.shape) == 4 else \
                self.net.layers[self.net.layers[layer_name].parents[0]].out_data[0].shape
            self.yolo_layer_params.update({layer_name: self.Params(self.net.layers[layer_name].params, shape[2])})


    @staticmethod
    def _resize_image(frame, size, keep_aspect_ratio=False):
        if not keep_aspect_ratio:
            resized_frame = cv2.resize(frame, size)
        else:
            # h, w = frame.shape[:2]
            # scale = min(size[1] / h, size[0] / w)
            # resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            iw, ih = frame.shape[0:2][::-1]
            w, h = size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            resized_frame = cv2.resize(frame, (nw, nh))
            # resized_frame = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
            # dx = (w - nw) // 2
            # dy = (h - nh) // 2
            # resized_frame[dy:dy + nh, dx:dx + nw, :] = frame
        return resized_frame

    def unify_inputs(self, inputs) -> dict:
        if not isinstance(inputs, dict):
            inputs_dict = {self.image_blob_name: inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def preprocess(self, inputs):
        img = self.resize_image(inputs[self.image_blob_name], (self.w, self.h))
        h, w = img.shape[:2]
        meta = {'original_shape': inputs[self.image_blob_name].shape,
                'resized_shape': img.shape}
        if h != self.h or w != self.w:
            resized_frame = np.full((self.h, self.w, 3), 128, dtype=np.uint8)
            dx = (self.w - w) // 2
            dy = (self.h - h) // 2
            resized_frame[dy:dy + h, dx:dx + w, :] = img
            img = resized_frame
            meta.update({'scales': (w / self.w, h / self.h)})
        if self.nchw_shape:
            img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img.reshape((self.n, self.c, self.h, self.w))
        inputs[self.image_blob_name] = img
        return inputs, meta

    @staticmethod
    def _parse_yolo_region(predictions, input_size, params, threshold):
        _, _, out_blob_h, out_blob_w = predictions.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                         "be equal to width. Current height = {}, current width = {}" \
                                         "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters ---------------------------------------
        objects = list()
        size_normalizer = input_size if params.isYoloV3 else (params.side, params.side)
        bbox_size = params.coords + 1 + params.classes
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for row, col, n in np.ndindex(params.side, params.side, params.num):
            # Getting raw values for each detection bounding box
            bbox = predictions[0, n * bbox_size:(n + 1) * bbox_size, row, col]
            x, y, width, height, object_probability = bbox[:5]
            class_probabilities = bbox[5:]
            if object_probability < threshold:
                continue
            # Process raw value
            x = (col + x) / params.side
            y = (row + y) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                width = np.exp(width)
                height = np.exp(height)
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            width = width * params.anchors[2 * n] / size_normalizer[0]
            height = height * params.anchors[2 * n + 1] / size_normalizer[1]

            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id] * object_probability
            if confidence < threshold:
                continue
            detection = Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                  confidence.item(), class_id.item())
            objects.append(detection)
        return objects

    @staticmethod
    def _filter(self, detections, iou_threshold):
        def iou(box_1, box_2):
            width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
            height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
            if width_of_overlap_area < 0 or height_of_overlap_area < 0:
                area_of_overlap = 0
            else:
                area_of_overlap = width_of_overlap_area * height_of_overlap_area
            box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
            box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
            area_of_union = box_1_area + box_2_area - area_of_overlap
            if area_of_union == 0:
                return 0
            return area_of_overlap / area_of_union

        detections = sorted(detections, key=lambda obj: obj.score, reverse=True)
        for i in range(len(detections)):
            if detections[i].score == 0:
                continue
            for j in range(i + 1, len(detections)):
                # We perform IOU only on objects of same class
                if detections[i].id != detections[j].id:
                     continue

                if iou(detections[i], detections[j]) > iou_threshold:
                    detections[j].score = 0

        return [det for det in detections if det.score > 0]

    @staticmethod
    def _resize_detections(detections, input_shape, original_image_shape, resized_image_shape, aspect_ratio_kept=False):
        if aspect_ratio_kept:
            scales = (resized_image_shape[0]/input_shape[0], resized_image_shape[1]/input_shape[1])
            detections = YOLO._resize_detections_letterbox(detections, scales)
        scales = (original_image_shape[0] , original_image_shape[1] )
        for detection in detections:
            detection.xmin *= scales[0]
            detection.xmax *= scales[0]
            detection.ymin *= scales[1]
            detection.ymax *= scales[1]
        return detections

    @staticmethod
    def _resize_detections_letterbox(detections, scales):
        offset = [0.5 * (1 - x) for x in scales]
        for detection in detections:
            detection.xmin = (detection.xmin - offset[0]) / scales[0]
            detection.xmax = (detection.xmax - offset[0]) / scales[0]
            detection.ymin = (detection.ymin - offset[1]) / scales[1]
            detection.ymax = (detection.ymax - offset[1]) / scales[1]
        return detections

    def postprocess(self, outputs, meta):
        detections = list()

        for layer_name, out_blob in outputs.items():
            out_blob = out_blob.reshape(self.net.layers[self.net.layers[layer_name].parents[0]].out_data[0].shape)
            layer_params = self.yolo_layer_params[layer_name]
            detections += self._parse_yolo_region(out_blob, meta['resized_shape'], layer_params, self.threshold)

        detections = self._filter(detections, self.iou_threshold)
        detections = self._resize_detections(detections, (self.w, self.h), meta['original_shape'][1::-1],
                                             meta['resized_shape'][1::-1], self.keep_aspect_ratio)

        return detections


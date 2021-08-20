from .image_model import ImageModel
from .utils import Detection, load_labels, clip_detections


class DetectionModel(ImageModel):
    def __init__(self, ie, model_path, input_transform=None, resize_type='default',
                 labels=None, threshold=None, iou_threshold=None):
        super().__init__(ie, model_path, input_transform=input_transform, resize_type=resize_type)
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        resized_shape = meta['resized_shape']
        original_shape = meta['original_shape']

        if self.resize_type=='letterbox':
            detections = self._resize_detections_letterbox(detections, original_shape[1::-1], resized_shape[1::-1])
        elif self.resize_type == 'keep_aspect_ratio':
            detections = self._resize_detections_with_aspect(detections, original_shape[1::-1], resized_shape[1::-1], (self.w, self.h))
        elif self.resize_type == 'default':
            detections = self._resize_detections(detections, original_shape[1::-1])
        else:
            raise RuntimeError('Unknown resize type {}'.format(self.resize_type))
        return clip_detections(detections, original_shape)

    def _parse_outputs(self, outputs, meta):
        raise NotImplementedError


    @staticmethod
    def _resize_detections(detections, original_image_size):
        '''
        original_shape - (w, h, ...)
        '''
        for detection in detections:
            detection.xmin *= original_image_size[0]
            detection.xmax *= original_image_size[0]
            detection.ymin *= original_image_size[1]
            detection.ymax *= original_image_size[1]
        return detections

    @staticmethod
    def _resize_detections_with_aspect(detections, original_image_size, resized_image_size, model_input_size):
        '''
        original_shape - (w, h)
        resized_shape - (w, h)
        model_shape - (w, h)
        '''
        print(model_input_size, resized_image_size)
        scale_x = model_input_size[0] / resized_image_size[0] * original_image_size[0]
        scale_y = model_input_size[1] / resized_image_size[1] * original_image_size[1]
        for detection in detections:
            detection.xmin *= scale_x
            detection.xmax *= scale_x
            detection.ymin *= scale_y
            detection.ymax *= scale_y
        return detections

    @staticmethod
    def _resize_detections_letterbox(detections, original_shape, resized_shape):
        scales = [x / y for x, y in zip(resized_shape, original_shape)]
        scale = min(scales)
        scales = (scale / scales[0], scale / scales[1])
        offset = [0.5 * (1 - x) for x in scales]
        for detection in detections:
            detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_shape[0]
            detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_shape[0]
            detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_shape[1]
            detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_shape[1]
        return detections

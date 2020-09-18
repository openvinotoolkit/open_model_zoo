from collections import namedtuple

import numpy as np

from .model import Model


Detection = namedtuple("Detection", "xmin ymin xmax ymax score id")


class SSD(Model):
    class SingleOutputParser:
        def __init__(self, all_outputs):
            if len(all_outputs) != 1:
                raise ValueError('Network must have only one output.')
            self.output_name, output_data = next(iter(all_outputs.items()))
            last_dim = np.shape(output_data)[-1]
            if last_dim != 7:
                raise ValueError('The last dimension of the output blob must be equal to 7, '
                                 'got {} instead.'.format(last_dim))

        def __call__(self, outputs):
            return [Detection(xmin, ymin, xmax, ymax, score, label)
                    for _, label, score, xmin, ymin, xmax, ymax in outputs[self.output_name][0][0]]

    class MultipleOutputParser:
        def __init__(self, layers, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
            def find_layer_by_name(name, layers):
                suitable_layers = [layer_name for layer_name in layers if name in layer_name]
                if not suitable_layers:
                    raise ValueError('Suitable layer for "{}" output is not found'.format(name))

                if len(suitable_layers) > 1:
                    raise ValueError('More than 1 layer matched to "{}" output'.format(name))

                return suitable_layers[0]
            self.labels_layer = find_layer_by_name(labels_layer, layers)
            self.scores_layer = find_layer_by_name(scores_layer, layers)
            self.bboxes_layer = find_layer_by_name(bboxes_layer, layers)

        def __call__(self, outputs):
            bboxes = outputs[self.bboxes_layer][0]
            scores = outputs[self.scores_layer][0]
            labels = outputs[self.labels_layer][0]
            return [Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]

    class BoxesLabelsParser:
        def __init__(self, input_size, all_outputs, labels_layer='labels', default_label=1):
            try:
                self.labels_layer = find_layer_by_name(labels_layer, all_outputs)
                log.info('Use output "{}" as the one containing labels of detected objects.'
                         .format(self.labels_layer))
            except ValueError:
                log.warning('Suitable layer for "{}" output is not found. '
                            'Treating detector as class agnostic with output label {}'
                            .format(labels_layer, default_label))
                self.labels_layer = None
                self.default_label = default_label

            self.input_size = input_size
            self.bboxes_layer = self.find_layer_bboxes_output(all_outputs)
            log.info('Use auto-detected output "{}" as the one containing detected bounding boxes.'
                     .format(self.bboxes_layer))

        @staticmethod
        def find_layer_bboxes_output(all_outputs):
            filter_outputs = [
                output_name for output_name, out_data in all_outputs.items()
                if len(np.shape(out_data)) == 2 and np.shape(out_data)[-1] == 5
            ]
            if not filter_outputs:
                raise ValueError('Suitable output with bounding boxes is not found')
            if len(filter_outputs) > 1:
                raise ValueError('More than 1 candidate for output with bounding boxes.')
            return filter_outputs[0]

        def __call__(self, outputs):
            bboxes = outputs[self.bboxes_layer]
            scores = bboxes[:, 4]
            bboxes = bboxes[:, :4]
            bboxes[:, 0::2] /= self.input_size[0]
            bboxes[:, 1::2] /= self.input_size[1]
            if self.labels_layer:
                labels = outputs[self.labels_layer] + 1
            else:
                labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)

            detections = [Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]
            return detections

    def __init__(self, *args, labels_map=None, keep_aspect_ratio_resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_aspect_ratio_resize = keep_aspect_ratio_resize
        self.labels_map = labels_map

        self.image_blob_name, self.image_info_blob_name = self._get_inputs(self.net)
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.n == 1, 'Only batch size == 1 is supported.'

        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def _get_inputs(self, net):
        image_blob_name = None
        image_info_blob_name = None
        for blob_name, blob in net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            elif len(blob.input_data.shape) == 2:
                image_info_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name, image_info_blob_name

    def _get_output_parser(self, net, image_blob_name, bboxes='bboxes', labels='labels', scores='scores'):
        try:
            parser = self.SingleOutputParser(net.outputs)
            self.logger.info('Use SingleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            parser = self.MultipleOutputParser(net.outputs, bboxes, scores, labels)
            self.logger.info('Use MultipleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            h, w = net.input_info[image_blob_name].input_data.shape[2:]
            parser = self.BoxesLabelsParser([w, h], net.outputs)
            self.logger.info('Use BoxesLabelsParser')
            return parser
        except ValueError:
            pass
        raise RuntimeError('Unsupported model outputs')

    @staticmethod
    def _resize_image(frame, size, keep_aspect_ratio=False):
        if not keep_aspect_ratio:
            resized_frame = cv2.resize(frame, size)
        else:
            h, w = frame.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return resized_frame

    def unify_inputs(self, inputs) -> dict:
        if not isinstance(inputs, dict):
            inputs_dict = {self.image_blob_name: inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def preprocess(self, inputs):
        img = self._resize_image(inputs[self.image_blob_name], (self.w, self.h), self.keep_aspect_ratio_resize)
        h, w = img.shape[:2]
        if self.image_info_blob_name is not None:
            inputs[self.image_info_blob_name] = [h, w, 1]
        meta = {'original_shape': inputs[self.image_blob_name].shape,
                'resized_shape': img.shape}
        if h != self.h or w != self.w:
            img = np.pad(img, ((0, self.h - h), (0, self.w - w), (0, 0)),
                         mode='constant', constant_values=0)
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img.reshape((self.n, self.c, self.h, self.w))
        inputs[self.image_blob_name] = img
        return inputs, meta

    def postprocess(self, outputs, meta):
        detections = self.output_parser(outputs)
        orginal_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']
        scale_x = self.w / resized_image_shape[1] * orginal_image_shape[1]
        scale_y = self.h / resized_image_shape[0] * orginal_image_shape[0]
        for detection in detections:
            detection.xmin *= scale_x
            detection.xmax *= scale_x
            detection.ymin *= scale_y
            detection.ymax *= scale_y
        return detections


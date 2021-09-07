"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from .model import Model
from .utils import Detection, resize_image, load_labels


class SSD(Model):
    def __init__(self, ie, model_path, input_transform, labels=None, keep_aspect_ratio_resize=False):
        super().__init__(ie, model_path, input_transform)

        self.keep_aspect_ratio_resize = keep_aspect_ratio_resize
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.image_blob_name, self.image_info_blob_name = self._get_inputs()
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape

        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def _get_inputs(self):
        image_blob_name = None
        image_info_blob_name = None
        for blob_name, blob in self.net.input_info.items():
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
            parser = SingleOutputParser(net.outputs)
            self.logger.info('Use SingleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            parser = MultipleOutputParser(net.outputs, bboxes, scores, labels)
            self.logger.info('Use MultipleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            parser = BoxesLabelsParser(net.outputs, net.input_info[image_blob_name].input_data.shape[2:][::-1])
            self.logger.info('Use BoxesLabelsParser')
            return parser
        except ValueError:
            pass
        raise RuntimeError('Unsupported model outputs')

    def preprocess(self, inputs):
        image = inputs

        resized_image = resize_image(image, (self.w, self.h), self.keep_aspect_ratio_resize)
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}

        h, w = resized_image.shape[:2]
        if h != self.h or w != self.w:
            resized_image = np.pad(resized_image, ((0, self.h - h), (0, self.w - w), (0, 0)),
                                   mode='constant', constant_values=0)
        resized_image = self.input_transform(resized_image)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: resized_image}
        if self.image_info_blob_name:
            dict_inputs[self.image_info_blob_name] = [self.h, self.w, 1]
        return dict_inputs, meta

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


def find_layer_by_name(name, layers):
    suitable_layers = [layer_name for layer_name in layers if name in layer_name]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found'.format(name))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output'.format(name))

    return suitable_layers[0]


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
        self.labels_layer = find_layer_by_name(labels_layer, layers)
        self.scores_layer = find_layer_by_name(scores_layer, layers)
        self.bboxes_layer = find_layer_by_name(bboxes_layer, layers)

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer][0]
        scores = outputs[self.scores_layer][0]
        labels = outputs[self.labels_layer][0]
        return [Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]


class BoxesLabelsParser:
    def __init__(self, layers, input_size, labels_layer='labels', default_label=0):
        try:
            self.labels_layer = find_layer_by_name(labels_layer, layers)
        except ValueError:
            self.labels_layer = None
            self.default_label = default_label

        self.bboxes_layer = self.find_layer_bboxes_output(layers)
        self.input_size = input_size

    @staticmethod
    def find_layer_bboxes_output(layers):
        filter_outputs = [name for name, data in layers.items() if len(np.shape(data)) == 2 and np.shape(data)[-1] == 5]
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
            labels = outputs[self.labels_layer]
        else:
            labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)

        detections = [Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]
        return detections

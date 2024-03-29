"""
 Copyright (c) 2020-2024 Intel Corporation
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

import cv2
import numpy as np

from types import SimpleNamespace as namespace

from .ie_tools import IEModel
from .segm_postprocess import postprocess


class MaskRCNN(IEModel):
    def __init__(self, core, model_path, labels_file, conf=.6, device='CPU'):
        super().__init__(core, model_path, labels_file, conf, device)

        self.input_keys = {'image'}
        self.output_keys = {'boxes', 'labels', 'masks'}
        self.input_keys_segmentoly = {'im_info', 'im_data'}
        self.output_keys_segmentoly = {'boxes', 'scores', 'classes', 'raw_masks'}

        self.segmentoly_type = self.check_segmentoly_type()
        self.input_tensor_name = 'im_data' if self.segmentoly_type else 'image'
        self.n, self.c, self.h, self.w = self.model.input(self.input_tensor_name).shape

    def check_segmentoly_type(self):
        for input_tensor_name in self.input_keys_segmentoly:
            try:
                self.model.input(input_tensor_name)
            except RuntimeError:
                return False
        for output_tensor_name in self.output_keys_segmentoly:
            try:
                self.model.output(output_tensor_name)
            except RuntimeError:
                return False
        return True

    def get_allowed_inputs_len(self):
        return (1, 2)

    def get_allowed_outputs_len(self):
        return (3, 4, 5)

    def _preprocess(self, frame):
        image_height, image_width = frame.shape[:2]
        scale = min(self.h / image_height, self.w / image_width)
        processed_image = cv2.resize(frame, None, fx=scale, fy=scale)
        processed_image = processed_image.astype('float32').transpose(2, 0, 1)
        height, width = processed_image.shape[1], processed_image.shape[2]
        im_info = np.array([height, width, 1.0], dtype='float32') if self.segmentoly_type else None
        meta=namespace(
            original_size=frame.shape[:2],
            processed_size=processed_image.shape[1:3],
        )
        return processed_image, im_info, meta

    def forward(self, im_data, im_info):
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should have the resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        feed_dict = {self.input_tensor_name: im_data}
        if im_info is not None:
            im_info = im_info.reshape(1, *im_info.shape)
            feed_dict['im_info'] = im_info
        self.infer_request.infer(feed_dict)
        if self.segmentoly_type:
            output = {name: self.infer_request.get_tensor(name).data[:] for name in self.output_keys_segmentoly}
            valid_detections_mask = output['classes'] > 0
            classes = output['classes'][valid_detections_mask]
            boxes = output['boxes'][valid_detections_mask]
            scores = output['scores'][valid_detections_mask]
            masks = output['raw_masks'][valid_detections_mask]
        else:
            output = {name: self.infer_request.get_tensor(name).data[:] for name in self.output_keys}
            valid_detections_mask = np.sum(output['boxes'], axis=1) > 0
            classes = output['labels'][valid_detections_mask] + 1
            boxes = output['boxes'][valid_detections_mask][:, :4]
            scores = output['boxes'][valid_detections_mask][:, 4]
            masks = output['masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=False):
        outputs = []
        for frame in frames:
            im_data, im_info, meta = self._preprocess(frame)

            boxes, classes, scores, _, masks = self.forward(im_data, im_info)
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta.original_size[0],
                                                        im_w=meta.original_size[1],
                                                        im_scale_y=meta.processed_size[0] / meta.original_size[0],
                                                        im_scale_x=meta.processed_size[1] / meta.original_size[1],
                                                        full_image_masks=True, encode_masks=False,
                                                        confidence_threshold=self.confidence,
                                                        segmentoly_postprocess=self.segmentoly_type)
            frame_output = []
            for i in range(len(scores)):
                if classes[i] in self.labels_to_hide:
                    bbox = [int(value) for value in boxes[i]]
                    if return_cropped_masks:
                        left, top, right, bottom = bbox
                        mask = masks[i][top:bottom, left:right]
                    else:
                        mask = masks[i]
                    frame_output.append([bbox, scores[i], mask])
            outputs.append(frame_output)
        return outputs


class SemanticSegmentation(IEModel):
    @staticmethod
    def set_classes_to_hide():
        return ('person', 'rider', )

    def get_detections(self, frames, only_class_person=True):
        outputs = []
        for frame in frames:
            out_h, out_w = frame.shape[:-1]
            res = self.forward(frame)
            output = []
            for data in res:
                data = data.transpose((1, 2, 0)).astype('uint8')
                data = cv2.resize(data, (out_w, out_h))
                data = np.isin(data, self.labels_to_hide).astype('uint8')
                output.append([[0, 0, out_w - 1, out_h - 1], 1., data.astype('uint8')])
            outputs.append(output)
        return outputs

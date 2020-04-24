"""
 Copyright (c) 2020 Intel Corporation
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

import json
import logging as log

import cv2
import numpy as np

from utils.ie_tools import IEModel
from .segm_postprocess import postprocess


class MaskRCNN(IEModel):
    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='', max_reqs=1):
        super().__init__(ie, model_path, conf, device, ext_path, max_reqs)

        required_input_keys = [{'im_info', 'im_data'}, {'im_data', 'im_info'}]
        current_input_keys = set(self.inputs_info.keys())
        assert current_input_keys in required_input_keys
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.outputs)

        self.n, self.c, self.h, self.w = self.inputs_info['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def _preprocess(self, frame):
        image_height, image_width = frame.shape[:2]
        scale = min(self.h / image_height, self.w / image_width)
        processed_image = cv2.resize(frame, None, fx=scale, fy=scale)
        processed_image = processed_image.astype('float32').transpose(2, 0, 1)

        sample = dict(original_image=frame,
                      meta=dict(original_size=frame.shape[:2],
                                processed_size=processed_image.shape[1:3]),
                      im_data=processed_image,
                      im_info=np.array([processed_image.shape[1], processed_image.shape[2], 1.0], dtype='float32'))
        return sample

    def forward(self, im_data, im_info):
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        im_info = im_info.reshape(1, *im_info.shape)
        output = self.net.infer(dict(im_data=im_data, im_info=im_info))

        classes = output['classes']
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = output['boxes'][valid_detections_mask]
        scores = output['scores'][valid_detections_mask]
        masks = output['raw_masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=False, only_class_person=True):
        outputs = []
        for frame in frames:
            data_batch = self._preprocess(frame)
            im_data = data_batch['im_data']
            im_info = data_batch['im_info']
            meta = data_batch['meta']

            boxes, classes, scores, _, masks = self.forward(im_data, im_info)
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta['original_size'][0],
                                                        im_w=meta['original_size'][1],
                                                        im_scale_y=meta['processed_size'][0] / meta['original_size'][0],
                                                        im_scale_x=meta['processed_size'][1] / meta['original_size'][1],
                                                        full_image_masks=True, encode_masks=False,
                                                        confidence_threshold=self.confidence)
            frame_output = []
            for i in range(len(scores)):
                if only_class_person and classes[i] != 1:
                    continue
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
    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='', max_reqs=1):
        super().__init__(ie, model_path, conf, device, ext_path, max_reqs)

    def get_detections(self, frames, only_class_person=True):
        outputs = []
        for frame in frames:
            out_h, out_w = frame.shape[:-1]
            res = self.forward(frame)
            output = []
            for batch, data in enumerate(res):
                data = data.transpose((1, 2, 0)).astype('uint8')
                data = cv2.resize(data, (out_w, out_h))
                data = np.where(data > 9, 1, 0)
                output.append([[0, 0, out_w - 1, out_h - 1], 1., data.astype('uint8')])
            outputs.append(output)
        return outputs
        

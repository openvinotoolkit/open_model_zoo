"""
 Copyright (c) 2019-2024 Intel Corporation
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
from abc import ABC, abstractmethod
from types import SimpleNamespace as namespace

import cv2
import numpy as np

from utils.ie_tools import IEModel
from .segm_postprocess import postprocess


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frames, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(IEModel, DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, core, model_path, trg_classes, conf=.6,
                 device='CPU', max_num_frames=1):
        super().__init__(core, model_path, device, 'Object Detection', max_num_frames)

        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames
        self.shapes = []
        for id, frame in enumerate(frames):
            self.shapes.append(frame.shape)
            self.forward_async(frame, id)

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, only_target_class):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            if only_target_class and detection[1] not in self.trg_classes:
                continue

            confidence = detection[2]
            if confidence < self.confidence:
                continue

            left = int(max(detection[3], 0) * frame_shape[1])
            top = int(max(detection[4], 0) * frame_shape[0])
            right = int(min(detection[5], 1) * frame_shape[1])
            bottom = int(min(detection[6], 1) * frame_shape[0])
            if self.expand_ratio != (1., 1.):
                w = (right - left)
                h = (bottom - top)
                dw = w * (self.expand_ratio[0] - 1.) / 2
                dh = h * (self.expand_ratio[1] - 1.) / 2
                left = max(int(left - dw), 0)
                right = int(right + dw)
                top = max(int(top - dh), 0)
                bottom = int(bottom + dh)

            detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


def _remove_batch_dim_from_each_element_because_outputs_itself_is_a_list_representing_a_batch(outputs):
    for idx, obj in enumerate(outputs):
        assert 1 == obj.shape[0]
        outputs[idx] = obj[0]


class VectorCNN(IEModel):
    """Wrapper class for a network returning a vector"""

    def __init__(self, core, model_path, device='CPU', max_reqs=100):
        self.max_reqs = max_reqs
        super().__init__(core, model_path, device, 'Object Reidentification', self.max_reqs)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for id, frame in enumerate(batch):
            super().forward_async(frame, id)
        outputs = self.grab_all_async()
        _remove_batch_dim_from_each_element_because_outputs_itself_is_a_list_representing_a_batch(outputs)
        return outputs

    def forward_async(self, batch):
        """Performs async forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            super().forward_async(frame)

    def wait_and_grab(self):
        outputs = self.grab_all_async()
        return outputs


class MaskRCNN(IEModel, DetectorInterface):
    """Wrapper class for a network returning masks of objects"""

    def __init__(self, core, model_path, trg_classes, conf=.6,
                 device='CPU', max_reqs=100):
        self.trg_classes = trg_classes
        self.max_reqs = max_reqs
        self.confidence = conf
        super().__init__(core, model_path, device, 'Instance Segmentation', self.max_reqs)

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

    def preprocess(self, frame):
        image_height, image_width = frame.shape[:2]
        scale = min(self.h / image_height, self.w / image_width)
        processed_image = cv2.resize(frame, None, fx=scale, fy=scale)
        processed_image = processed_image.astype('float32').transpose(2, 0, 1)

        return namespace(
            original_image=frame,
            meta=namespace(
                original_size=frame.shape[:2],
                processed_size=processed_image.shape[1:3],
            ),
            im_data=processed_image,
            im_info=np.array([processed_image.shape[1], processed_image.shape[2], 1.0], dtype='float32'),
        )

    def forward(self, im_data, im_info):
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        feed_dict = {self.input_tensor_name: im_data}
        if im_info is not None:
            im_info = im_info.reshape(1, *im_info.shape)
            feed_dict['im_info'] = im_info
        self.infer_queue[0].infer(feed_dict)
        if self.segmentoly_type:
            output = {name: self.infer_queue[0].get_tensor(name).data for name in self.output_keys_segmentoly}
            valid_detections_mask = output['classes'] > 0
            classes = output['classes'][valid_detections_mask]
            boxes = output['boxes'][valid_detections_mask]
            scores = output['scores'][valid_detections_mask]
            masks = output['raw_masks'][valid_detections_mask]
        else:
            output = {name: self.infer_queue[0].get_tensor(name).data for name in self.output_keys}
            valid_detections_mask = np.sum(output['boxes'], axis=1) > 0
            classes = output['labels'][valid_detections_mask] + 1
            boxes = output['boxes'][valid_detections_mask][:, :4]
            scores = output['boxes'][valid_detections_mask][:, 4]
            masks = output['masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=True, only_target_class=True):
        outputs = []
        for frame in frames:
            data_batch = self.preprocess(frame)
            im_data = data_batch.im_data
            im_info = data_batch.im_info if self.segmentoly_type else None
            meta = data_batch.meta

            boxes, classes, scores, _, masks = self.forward(im_data, im_info)
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta.original_size[0],
                                                        im_w=meta.original_size[1],
                                                        im_scale_y=meta.processed_size[0] / meta.original_size[0],
                                                        im_scale_x=meta.processed_size[1] / meta.original_size[1],
                                                        full_image_masks=True, encode_masks=False,
                                                        confidence_threshold=self.confidence,
                                                        segmentoly_type=self.segmentoly_type)
            frame_output = []
            for i in range(len(scores)):
                if only_target_class and classes[i] not in self.trg_classes:
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

    def run_async(self, frames, index):
        self.frames = frames

    def wait_and_grab(self):
        return self.get_detections(self.frames)


class DetectionsFromFileReader(DetectorInterface):
    """Read detection from *.json file.
    Format of the file should be:
    [
        {'frame_id': N,
         'scores': [score0, score1, ...],
         'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]},
        ...
    ]
    """

    def __init__(self, input_file, score_thresh):
        self.input_file = input_file
        self.score_thresh = score_thresh
        self.detections = []
        log.info('Loading {}'.format(input_file))
        with open(input_file) as f:
            all_detections = json.load(f)
        for source_detections in all_detections:
            detections_dict = {}
            for det in source_detections:
                detections_dict[det['frame_id']] = {'boxes': det['boxes'], 'scores': det['scores']}
            self.detections.append(detections_dict)

    def run_async(self, frames, index):
        self.last_index = index

    def wait_and_grab(self):
        output = []
        for source in self.detections:
            valid_detections = []
            if self.last_index in source:
                for bbox, score in zip(source[self.last_index]['boxes'], source[self.last_index]['scores']):
                    if score > self.score_thresh:
                        bbox = [int(value) for value in bbox]
                        valid_detections.append((bbox, score))
            output.append(valid_detections)
        return output

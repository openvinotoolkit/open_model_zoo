"""
 Copyright (c) 2019 Intel Corporation
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

import glog
import json
from collections import namedtuple

import cv2
import numpy as np

from utils.ie_tools import load_ie_model
from .segm_postrocess import postprocess


class Detector:
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='', max_num_frames=1):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames)
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def get_detections(self, frames):
        """Returns all detections on frames"""
        assert len(frames) <= self.max_num_frames

        all_detections = []
        for i in range(len(frames)):
            self.net.forward_async(frames[i])
        outputs = self.net.grab_all_async()

        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, frames[i].shape)
            all_detections.append(detections)

        return all_detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
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


class VectorCNN:
    """Wrapper class for a network returning a vector"""

    def __init__(self, ie, model_path, device='CPU', ext_path='', max_reqs=100):
        self.max_reqs = max_reqs
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=self.max_reqs)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)
        outputs = self.net.grab_all_async()
        return outputs

    def forward_async(self, batch):
        """Performs async forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)

    def wait_and_grab(self):
        outputs = self.net.grab_all_async()
        return outputs


ReidFeature = namedtuple('ReidFeature', 'f o')


class ReIDWithOrientationWrapper:
    def __init__(self, reid, orientation_classifier=None, cl_threshold=0.5):
        self.reid = reid
        self.orientation_classifier = orientation_classifier
        self.cl_threshold = cl_threshold

    def forward(self, rois):
        self.reid.forward_async(rois)
        if self.orientation_classifier is not None:
            self.orientation_classifier.forward_async(rois)
            embeddings = self.reid.wait_and_grab()
            orientations = self.orientation_classifier.wait_and_grab()
            for i, vec in enumerate(orientations):
                idx = np.argmax(vec)
                conf = vec[0][idx]
                if conf < self.cl_threshold:
                    idx = -1
                embeddings[i] = ReidFeature(embeddings[i], idx)
        else:
            embeddings = self.reid.wait_and_grab()
            for i, vec in enumerate(embeddings):
                embeddings[i] = ReidFeature(embeddings[i], -1)

        return embeddings


class MaskRCNN:
    """Wrapper class for a network returning masks of objects"""

    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='',
                 max_reqs=100, mean_pixel=(102.9801, 115.9465, 122.7717)):
        self.max_reqs = max_reqs
        self.confidence = conf
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=self.max_reqs)

        required_input_keys = [{'im_info', 'im_data'}, {'im_data', 'im_info'}]
        current_input_keys = set(self.net.inputs_info.keys())
        assert current_input_keys in required_input_keys
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.net.outputs)

        self.n, self.c, self.h, self.w = self.net.inputs_info['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

        self.transforms = self.Compose(
            [
                self.Resize(max_size=None, window_size=(self.h, self.w), size=None),
                self.Normalize(mean=mean_pixel, std=[1., 1., 1.]),
            ]
        )

    def preprocess(self, frame):
        processed_image = self.transforms({'image': frame})['image']
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
        output = self.net.net.infer(dict(im_data=im_data, im_info=im_info))

        classes = output['classes']
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = output['boxes'][valid_detections_mask]
        scores = output['scores'][valid_detections_mask]
        masks = output['raw_masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=True, only_class_person=True):
        outputs = []
        for frame in frames:
            data_batch = self.preprocess(frame)
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

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, sample):
            for t in self.transforms:
                sample = t(sample)
            return sample

    class Resize(object):
        def __init__(self, max_size=None, window_size=None, size=None):
            super().__init__()
            assert int(max_size is not None) + int(window_size is not None) + int(size is not None) == 1

            self.short_side_max = None
            self.long_side_max = None
            if max_size is not None:
                self.short_side_max, self.long_side_max = max_size

            self.height_max = None
            self.width_max = None
            if window_size is not None:
                self.height_max, self.width_max = window_size

            self.height = None
            self.width = None
            if size is not None:
                self.height, self.width = size

        def get_scale(self, image_size):
            if self.height is not None:
                scale_x, scale_y = self.width / image_size[1], self.height / image_size[0]
            elif self.height_max is not None:
                im_scale = min(self.height_max / image_size[0], self.width_max / image_size[1])
                scale_x, scale_y = im_scale, im_scale
            else:
                im_scale = min(self.short_side_max / min(image_size), self.long_side_max / max(image_size))
                scale_x, scale_y = im_scale, im_scale
            return scale_x, scale_y

        def __call__(self, sample):
            image_size = sample['image'].shape[:2]
            scale_x, scale_y = self.get_scale(image_size)

            # Resize image.
            sample['image'] = cv2.resize(sample['image'], None, fx=scale_x, fy=scale_y)
            h, w = sample['image'].shape[:2]

            # Resize boxes.
            if 'gt_boxes' in sample:
                sample['gt_boxes'] *= [scale_x, scale_y, scale_x, scale_y]
                sample['gt_boxes'] = np.clip(sample['gt_boxes'], 0, [w - 1, h - 1, w - 1, h - 1])

            # Resize masks.
            if 'gt_masks' in sample:
                sample['gt_masks'] = [[np.clip(part * [scale_x, scale_y], 0, [w - 1, h - 1]) for part in obj]
                                      for obj in sample['gt_masks']]

            return sample

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, sample):
            sample['image'] = sample['image'].astype('float32').transpose(2, 0, 1)
            sample['image'][:, :, 0] = (sample['image'][:, :, 0] - self.mean[0]) / self.std[0]
            sample['image'][:, :, 1] = (sample['image'][:, :, 1] - self.mean[1]) / self.std[1]
            sample['image'][:, :, 2] = (sample['image'][:, :, 2] - self.mean[2]) / self.std[2]
            return sample


class DetectionsFromFileReader(object):
    """Read detection from *.json file.
    Format of the file should be:
    [
        {'frame_id': N,
         'scores': [score0, score1, ...],
         'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]},
        ...
    ]
    """
    def __init__(self, input_files, score_thresh):
        self.input_files = input_files
        self.score_thresh = score_thresh
        self.detections = []
        for input_file in input_files:
            glog.info('Loading {}'.format(input_file))
            with open(input_file) as f:
                detections = json.load(f)
                detections_dict = {}
                for det in detections:
                    detections_dict[det['frame_id']] = {'boxes': det['boxes'], 'scores': det['scores']}
                self.detections.append(detections_dict)

    def get_detections(self, frame_id):
        output = []
        for source in self.detections:
            valid_detections = []
            if frame_id in source:
                for bbox, score in zip(source[frame_id]['boxes'], source[frame_id]['scores']):
                    if score > self.score_thresh:
                        bbox = [int(value) for value in bbox]
                        valid_detections.append((bbox, score))
            output.append(valid_detections)
        return output

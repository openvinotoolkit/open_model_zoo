"""
 Copyright (c) 2018-2021 Intel Corporation

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

from utils import cut_rois, resize_input
from ie_module import Module


class FaceIdentifier(Module):
    # Taken from the description of the model:
    # intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, ie, model, match_threshold=0.5, match_algo='HUNGARIAN'):
        super(FaceIdentifier, self).__init__(ie, model)

        assert len(self.model.input_info) == 1, 'Expected 1 input blob'
        assert len(self.model.outputs) == 1, 'Expected 1 output blob'

        self.input_blob = next(iter(self.model.input_info))
        self.output_blob = next(iter(self.model.outputs))
        self.input_shape = self.model.input_info[self.input_blob].input_data.shape
        output_shape = self.model.outputs[self.output_blob].shape

        assert len(output_shape) in (2, 4), \
            'Expected model output shape [1, n, 1, 1] or [1, n], got {}'.format(output_shape)

        self.faces_database = None
        self.match_threshold = match_threshold
        self.match_algo = match_algo

    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        image = frame.copy()
        inputs = cut_rois(image, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(FaceIdentifier, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_threshold(self):
        return self.match_threshold

    def postprocess(self):
        descriptors = self.get_descriptors()

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors, self.match_algo)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)

            results.append(self.Result(id, distance, descriptors[num]))
        return results, unknowns_list

    def get_descriptors(self):
        return [out[self.output_blob].buffer.flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=0)
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            'Input lengths differ, got {} and {}'.format(len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            scale = np.array((image.shape[1], image.shape[0]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=np.float64) * scale
            landmarks = image_landmarks * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            cv2.warpAffine(image, transform, tuple(scale), image, flags=cv2.WARP_INVERSE_MAP)

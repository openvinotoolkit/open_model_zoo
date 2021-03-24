"""
 Copyright (C) 2020-2021 Intel Corporation

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

from .hpe_associative_embedding import AssociativeEmbeddingDecoder
from .model import Model
from .utils import resize_image


class HpeHRNet(Model):
    def __init__(self, ie, model_path, target_size, aspect_ratio, prob_threshold, size_divisor=32):
        super().__init__(ie, model_path)
        self.image_blob_name = self._get_inputs(self.net)
        self.heatmaps_blob_name = find_layer_by_name('heatmaps', self.net.outputs)
        try:
            self.nms_heatmaps_blob_name = find_layer_by_name('nms_heatmaps', self.net.outputs)
        except ValueError:
            self.nms_heatmaps_blob_name = self.heatmaps_blob_name
        self.embeddings_blob_name = find_layer_by_name('embeddings', self.net.outputs)

        self.output_scale = self.net.input_info[self.image_blob_name].input_data.shape[-1] / self.net.outputs[self.heatmaps_blob_name].shape[-1]

        if target_size is None:
            h, w = self.net.input_info[self.image_blob_name].input_data.shape[-2:]
            target_size = min(h, w)
        if aspect_ratio >= 1.0:  # img width >= height
            input_height, input_width = target_size, round(target_size * aspect_ratio)
        else:
            input_height, input_width = round(target_size / aspect_ratio), target_size
        self.h = (input_height + size_divisor - 1) // size_divisor * size_divisor
        self.w = (input_width + size_divisor - 1) // size_divisor * size_divisor
        default_input_shape = self.net.input_info[self.image_blob_name].input_data.shape
        input_shape = {self.image_blob_name: (default_input_shape[:-2] + [self.h, self.w])}
        self.logger.info('Reshape net to {}'.format(input_shape))
        self.net.reshape(input_shape)

        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=self.net.outputs[self.heatmaps_blob_name].shape[1],
            adjust=True,
            refine=True,
            delta=0.5,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1,
            pose_threshold=prob_threshold,
            use_detection_val=True,
            ignore_too_much=False,
            dist_reweight=True)
        self.size_divisor = size_divisor

    @staticmethod
    def _get_inputs(net):
        image_blob_name = None
        for blob_name, blob in net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name

    def preprocess(self, inputs):
        img = resize_image(inputs, (self.w, self.h), keep_aspect_ratio=True)
        h, w = img.shape[:2]
        if not (self.h - self.size_divisor < h <= self.h and self.w - self.size_divisor < w <= self.w):
            self.logger.warn("Chosen model aspect ratio doesn't match image aspect ratio")
        resize_img_scale = np.array((inputs.shape[1] / w, inputs.shape[0] / h), np.float32)

        pad = [(self.h - h + 1) // 2, (self.w - w + 1) // 2, (self.h - h) // 2, (self.w - w) // 2]
        img = np.pad(img, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
                     mode='constant', constant_values=0)
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img[None]
        meta = {
            'original_size': inputs.shape[:2],
            'resize_img_scale': resize_img_scale
        }
        return {self.image_blob_name: img}, meta

    def postprocess(self, outputs, meta):
        heatmaps = outputs[self.heatmaps_blob_name]
        nms_heatmaps = outputs[self.nms_heatmaps_blob_name]
        aembds = outputs[self.embeddings_blob_name]

        poses, scores = self.decoder(heatmaps, aembds, nms_heatmaps=nms_heatmaps)
        # Rescale poses to the original image.
        poses = self.transform_preds(poses, meta)
        return poses, scores

    def transform_preds(self, poses, meta):
        # compute trans
        scale = min(meta['resize_img_scale'])
        trans = np.zeros((2, 3))
        trans[0][0] = trans[1][1] = scale * self.output_scale

        shift = (max(meta['original_size']) - max(self.h, self.w) * scale) / 2
        trans[np.argmax(meta['original_size'][::-1])][2] = shift

        # affine_trans
        for person in poses:
            for joint in person:
                joint[:2] = self.affine_transform(joint[:2], trans)
        return poses

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.])
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


def find_layer_by_name(name, layers):
    suitable_layers = [layer_name for layer_name in layers if layer_name.startswith(name)]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found'.format(name))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output'.format(name))

    return suitable_layers[0]

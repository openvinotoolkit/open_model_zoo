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

from collections import defaultdict

import cv2
import numpy as np


from ..adapters import Adapter
from ..config import ConfigValidator, StringField, NumberField, BoolField, ConfigError
from ..representation import TextDetectionPrediction, CharacterRecognitionPrediction


class TextDetectionAdapter(Adapter):
    __provider__ = 'text_detection'
    prediction_types = (TextDetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'pixel_link_out': StringField(
                description="Name of layer containing information related to linkage "
                            "between pixels and their neighbors."
            ),
            'pixel_class_out': StringField(
                description="Name of layer containing information related to "
                            "text/no-text classification for each pixel."
            ),
            'pixel_class_confidence_threshold': NumberField(
                description='confidence threshold for valid segmentation mask',
                optional=True, default=0.8, value_type=float, min_value=0, max_value=1
            ),
            'pixel_link_confidence_threshold': NumberField(
                description='confidence threshold for valid pixel links',
                optional=True, default=0.8, value_type=float, min_value=0, max_value=1
            ),
            'min_area': NumberField(
                value_type=int, min_value=0, default=0, optional=True,
                description='minimal area for valid text prediction'
            ),
            'min_height': NumberField(
                value_type=int, min_value=0, default=0, optional=True,
                description='minimal height for valid text prediction'
            )
        })

        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.pixel_link_out = self.get_value_from_config('pixel_link_out')
        self.pixel_class_out = self.get_value_from_config('pixel_class_out')
        self.pixel_link_confidence_threshold = self.get_value_from_config('pixel_link_confidence_threshold')
        self.pixel_class_confidence_threshold = self.get_value_from_config('pixel_class_confidence_threshold')
        self.min_area = self.get_value_from_config('min_area')
        self.min_height = self.get_value_from_config('min_height')

    def process(self, raw, identifiers=None, frame_meta=None):
        results = []
        predictions = self._extract_predictions(raw, frame_meta)

        def _input_parameters(input_meta):
            input_shape = next(iter(input_meta.get('input_shape').values()))
            original_image_size = input_meta.get('image_size')
            layout = 'NCHW' if input_shape[1] == original_image_size[2] else 'NHWC'

            return original_image_size, layout
        raw_output = zip(identifiers, frame_meta, predictions[self.pixel_link_out], predictions[self.pixel_class_out])
        for identifier, current_frame_meta, link_data, cls_data in raw_output:
            image_size, layout = _input_parameters(current_frame_meta)
            if layout == 'NCHW':
                link_data = link_data.transpose((1, 2, 0))
                cls_data = cls_data.transpose((1, 2, 0))
            new_link_data = link_data.reshape([*link_data.shape[:2], 8, 2])
            new_link_data = self.softmax(new_link_data)
            cls_data = self.softmax(cls_data)
            decoded_rects = self.to_boxes(image_size, cls_data[:, :, 1], new_link_data[:, :, :, 1])
            results.append(TextDetectionPrediction(identifier, decoded_rects))

        return results

    def mask_to_bboxes(self, mask, image_shape):
        """ Converts mask to bounding boxes. """

        def rect_to_xys(rect, image_shape):
            """ Converts rotated rectangle to points. """

            height, width = image_shape[0:2]

            def get_valid_x(x_coord):
                return np.clip(x_coord, 0, width - 1)

            def get_valid_y(y_coord):
                return np.clip(y_coord, 0, height - 1)

            rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
            points = cv2.boxPoints(rect)
            points = points.astype(np.int0)
            for i_xy, (x_coord, y_coord) in enumerate(points):
                x_coord = get_valid_x(x_coord)
                y_coord = get_valid_y(y_coord)
                points[i_xy, :] = [x_coord, y_coord]

            return points

        def min_area_rect(contour):
            """ Returns minimum area rectangle. """

            (center_x, center_y), (width, height), theta = cv2.minAreaRect(contour)
            return [center_x, center_y, width, height, theta], width * height

        image_h, image_w = image_shape[0:2]

        bboxes = []
        max_bbox_idx = mask.max()
        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        for bbox_idx in range(1, max_bbox_idx + 1):
            bbox_mask = (mask == bbox_idx).astype(np.uint8)
            cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if np.size(cnts) == 0:
                continue
            cnt = cnts[0]
            rect, rect_area = min_area_rect(cnt)

            box_width, box_height = rect[2:-1]
            if min(box_width, box_height) < self.min_height:
                continue

            if rect_area < self.min_area:
                continue

            xys = rect_to_xys(rect, image_shape)
            bboxes.append(xys)

        return bboxes

    @staticmethod
    def softmax(logits):
        """ Returns softmax given logits. """

        max_logits = np.max(logits, axis=-1, keepdims=True)
        numerator = np.exp(logits - max_logits)
        denominator = np.sum(numerator, axis=-1, keepdims=True)

        return numerator / denominator

    def to_boxes(self, image_shape, segm_pos_scores, link_pos_scores):
        """ Returns boxes for each image in batch. """

        mask = self.decode_image(segm_pos_scores, link_pos_scores)
        mask = np.asarray(mask, np.int32)[...]
        bboxes = self.mask_to_bboxes(mask, image_shape)

        return bboxes

    def decode_image(self, segm_scores, link_scores):
        """ Convert softmax scores to mask. """

        segm_mask = segm_scores >= self.pixel_class_confidence_threshold
        link_mask = link_scores >= self.pixel_link_confidence_threshold
        points = list(zip(*np.where(segm_mask)))
        height, width = np.shape(segm_mask)
        group_mask = dict.fromkeys(points, -1)

        def find_parent(point):
            return group_mask[point]

        def set_parent(point, parent):
            group_mask[point] = parent

        def is_root(point):
            return find_parent(point) == -1

        def find_root(point):
            root = point
            update_parent = False
            while not is_root(root):
                root = find_parent(root)
                update_parent = True

            if update_parent:
                set_parent(point, root)

            return root

        def join(point1, point2):
            root1 = find_root(point1)
            root2 = find_root(point2)

            if root1 != root2:
                set_parent(root1, root2)

        def get_neighbours(x_coord, y_coord):
            """ Returns 8-point neighbourhood of given point. """

            return [
                (x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1),
                (x_coord - 1, y_coord), (x_coord + 1, y_coord),
                (x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)
            ]

        def is_valid_coord(x_coord, y_coord, width, height):
            """ Returns true if given point inside image frame. """
            return 0 <= x_coord < width and 0 <= y_coord < height

        def get_all():
            root_map = {}

            def get_index(root):
                if root not in root_map:
                    root_map[root] = len(root_map) + 1
                return root_map[root]

            mask = np.zeros_like(segm_mask, dtype=np.int32)
            for point in points:
                point_root = find_root(point)
                bbox_idx = get_index(point_root)
                mask[point] = bbox_idx
            return mask

        for point in points:
            y_coord, x_coord = point
            neighbours = get_neighbours(x_coord, y_coord)
            for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
                if is_valid_coord(neighbour_x, neighbour_y, width, height):
                    link_value = link_mask[y_coord, x_coord, n_idx]
                    segm_value = segm_mask[neighbour_y, neighbour_x]
                    if link_value and segm_value:
                        join(point, (neighbour_y, neighbour_x))

        mask = get_all()
        return mask


class LPRAdapter(Adapter):
    __provider__ = 'lpr'
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        if not self.label_map:
            raise ConfigError('LPR adapter requires dataset label map for correct decoding.')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        predictions = raw_output[self.output_blob]
        result = []
        for identifier, output in zip(identifiers, predictions):
            decoded_out = self.decode(output.reshape(-1))
            result.append(CharacterRecognitionPrediction(identifier, decoded_out))

        return result

    def decode(self, outputs):
        decode_out = str()
        for output in outputs:
            if output == -1:
                break
            decode_out += str(self.label_map[int(output)])

        return decode_out


class BeamSearchDecoder(Adapter):
    __provider__ = 'beam_search_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'beam_size': NumberField(
                optional=True, value_type=int, min_value=1, default=10,
                description="Size of the beam to use during decoding."
            ),
            'blank_label': NumberField(
                optional=True, value_type=int, min_value=0, description="Index of the CTC blank label."
            ),
            'softmaxed_probabilities': BoolField(
                optional=True, default=False, description="Indicator that model uses softmax for output layer "
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')

        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label', len(self.label_map))
        self.softmaxed_probabilities = self.get_value_from_config('softmaxed_probabilities')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        output = np.swapaxes(output, 0, 1)

        result = []
        for identifier, data in zip(identifiers, output):
            if self.softmaxed_probabilities:
                data = np.log(data)
            seq = self.decode(data, self.beam_size, self.blank_label)
            decoded = ''.join(str(self.label_map[char]) for char in seq)
            result.append(CharacterRecognitionPrediction(identifier, decoded))
        return result

    @staticmethod
    def decode(probabilities, beam_size=10, blank_id=None):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            probabilities: The output log probabilities for each time step.
            Should be an array of shape (time x output dim).
            beam_size (int): Size of the beam to use during decoding.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        def make_new_beam():
            return defaultdict(lambda: (-np.inf, -np.inf))

        def log_sum_exp(*args):
            if all(a == -np.inf for a in args):
                return -np.inf
            a_max = np.max(args)
            lsp = np.log(np.sum(np.exp(a - a_max) for a in args))

            return a_max + lsp

        times, symbols = probabilities.shape
        # Initialize the beam with the empty sequence, a probability of 1 for ending in blank
        # and zero for ending in non-blank (in log space).
        beam = [(tuple(), (0.0, -np.inf))]

        for time in range(times):
            # A default dictionary to store the next step candidates.
            next_beam = make_new_beam()

            for symbol_id in range(symbols):
                current_prob = probabilities[time, symbol_id]

                for prefix, (prob_blank, prob_non_blank) in beam:
                    # If propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if symbol_id == blank_id:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_blank = log_sum_exp(
                            next_prob_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)
                        continue
                    # Extend the prefix by the new character symbol and add it to the beam.
                    # Only the probability of not ending in blank gets updated.
                    end_t = prefix[-1] if prefix else None
                    next_prefix = prefix + (symbol_id,)
                    next_prob_blank, next_prob_non_blank = next_beam[next_prefix]
                    if symbol_id != end_t:
                        next_prob_non_blank = log_sum_exp(
                            next_prob_non_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                    else:
                        # Don't include the previous probability of not ending in blank (prob_non_blank) if symbol
                        #  is repeated at the end. The CTC algorithm merges characters not separated by a blank.
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_blank + current_prob)

                    next_beam[next_prefix] = (next_prob_blank, next_prob_non_blank)
                    # If symbol is repeated at the end also update the unchanged prefix. This is the merging case.
                    if symbol_id == end_t:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_non_blank + current_prob)
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)

            beam = sorted(next_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)[:beam_size]

        best = beam[0]

        return best[0]

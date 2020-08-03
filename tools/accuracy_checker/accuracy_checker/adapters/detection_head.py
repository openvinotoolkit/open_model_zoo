"""
Copyright (c) 2018-2020 Intel Corporation

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
from collections import namedtuple

from ..adapters import Adapter
from ..config import NumberField, ListField, ConfigError
from ..representation import DetectionPrediction

DetectionLayerOutput = namedtuple(
    'DetectionLayerOutput',
    ['prob_name', 'reg_name', 'anchor_index', 'anchor_size', 'win_scale', 'win_length', 'win_trans_x', 'win_trans_y']
)

class HeadDetectionAdapter(Adapter):
    __provider__ = 'head_detection'
    predcition_types = (DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'score_threshold': NumberField(
                value_type=float, min_value=0, max_value=1, default=0.35, optional=True,
                description='Score threshold value used to discern whether a face is valid'),
            'anchor_sizes': ListField(
                value_type=int, optional=False,
                description='Anchor sizes for each base output layer'),
            'window_scales': ListField(
                value_type=int, optional=False,
                description='Window scales for each base output layer'),
            'window_lengths': ListField(
                value_type=int, optional=False,
                description='Window lenghts for each base output layer')
        })
        return parameters

    def configure(self):
        self.score_threshold = self.get_value_from_config('score_threshold')
        self.layer_info = {
            'anchor_sizes': self.get_value_from_config('anchor_sizes'),
            'window_scales': self.get_value_from_config('window_scales'),
            'window_lengths': self.get_value_from_config('window_lengths')
        }
        if len({len(x) for x in self.layer_info.values()}) != 1:
            raise ConfigError('There must be equal number of layer names, anchor sizes, '
                              'window scales, and window sizes')
        self.output_layers = self.generate_output_layer_info()

    def generate_output_layer_info(self):
        output_layers = []

        for i in range(len(self.layer_info['anchor_sizes'])):
            start = 1.5
            anchor_size = self.layer_info['anchor_sizes'][i]
            window_scale = self.layer_info['window_scales'][i]
            window_length = self.layer_info['window_lengths'][i]
            if anchor_size % 3 == 0:
                start = -anchor_size / 3.0
            elif anchor_size % 2 == 0:
                start = -anchor_size / 2.0 + 0.5
            k = 1
            for row in range(anchor_size):
                for col in range(anchor_size):
                    out_layer = DetectionLayerOutput(
                        prob_name="out_prob",
                        reg_name="out_reg",
                        anchor_index=k - 1,
                        anchor_size=anchor_size * anchor_size,
                        win_scale=window_scale,
                        win_length=window_length,
                        win_trans_x=float((start + col) / anchor_size),
                        win_trans_y=float((start + row) / anchor_size)
                    )
                    output_layers.append(out_layer)
                    k += 1
        return output_layers

    def process(self, raw, identifiers=None, frame_meta=None):
        _, _, network_h, network_w = frame_meta[0]['input_shape']['images']
        scale_factor_w = 1 / frame_meta[0]['scale_x']
        scale_factor_h = 1 / frame_meta[0]['scale_y']
        base_prob_idx = 0
        base_reg_idx = 0
        result = []
        for batch_index, identifier in enumerate(identifiers):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            for layer in self.output_layers:
                output_width = int(network_w / layer.win_scale)
                output_height = int(network_h / layer.win_scale)
                if network_w % layer.win_scale != 0:
                    output_width += 1
                if network_h % layer.win_scale != 0:
                    output_height += 1
                prob_arr = raw[batch_index][layer.prob_name][0]
                reg_arr = raw[batch_index][layer.reg_name][0]
                for y in range(output_height):
                    for x in range(output_width):
                        score = prob_arr[base_prob_idx + y * output_width + x][1]
                        if score >= self.score_threshold:
                            candidate_x = (x + layer.win_trans_x) * layer.win_scale + layer.win_scale / 2.0
                            candidate_y = (y + layer.win_trans_y) * layer.win_scale + layer.win_scale / 2.0
                            candidate_width = layer.win_length
                            candidate_height = layer.win_length

                            reg_x = reg_arr[base_reg_idx + y * output_width + x][0]
                            reg_y = reg_arr[base_reg_idx + y * output_width + x][1]
                            reg_width = reg_arr[base_reg_idx + y * output_width + x][2]
                            reg_height = reg_arr[base_reg_idx + y * output_width + x][3]

                            candidate_x += ((reg_x - 0.5) * layer.win_length)
                            candidate_y += ((reg_y - 0.5) * layer.win_length)
                            candidate_width *= (reg_width + 0.5)
                            candidate_height *= (reg_height + 0.5)

                            width = scale_factor_w * candidate_width
                            height = scale_factor_h * candidate_height
                            x_min = scale_factor_w * candidate_x - width / 2.0
                            y_min = scale_factor_h * candidate_y - height / 2.0
                            width += x_min - int(x_min)
                            height += y_min - int(y_min)

                            detections['x_mins'].append(x_min)
                            detections['y_mins'].append(y_min)
                            detections['x_maxs'].append(x_min + width)
                            detections['y_maxs'].append(y_min + height)
                            detections['scores'].append(score)

                base_prob_idx += output_width * output_height
                base_reg_idx += output_width * output_height

            result.append(
                DetectionPrediction(
                    identifier=identifier,
                    x_mins=detections['x_mins'],
                    y_mins=detections['y_mins'],
                    x_maxs=detections['x_maxs'],
                    y_maxs=detections['y_maxs'],
                    scores=detections['scores']
                )
            )

        return result

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

from ..config import BoolField
from ..representation import DetectionPrediction, ClassificationPrediction
from ..adapters import Adapter


class XML2DetectionAdapter(Adapter):
    """
    Class for converting xml detection results in OpenCV FileStorage format to DetectionPrediction representation.
    """

    __provider__ = 'xml_detection'
    prediction_types = (DetectionPrediction, )

    def process(self, tree, identifiers=None, frame_meta=None):
        class_to_ind = dict(zip(self.label_map.values(), range(len(self.label_map.values()))))

        result = {}
        for frames in tree.getroot():
            for frame in frames:
                identifier = frame.tag + '.png'
                labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
                for prediction in frame:
                    if prediction.find('is_ignored'):
                        continue

                    label = prediction.find('type')
                    if not label:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('type'))
                    label = class_to_ind[label.text]

                    confidence = prediction.find('confidence')
                    if confidence is None:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('confidence'))
                    confidence = float(confidence.text)

                    box = prediction.find('roi')
                    if not box:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('roi'))
                    box = list(map(float, box.text.split()))

                    labels.append(label)
                    scores.append(confidence)
                    x_mins.append(box[0])
                    y_mins.append(box[1])
                    x_maxs.append(box[0] + box[2])
                    y_maxs.append(box[1] + box[3])

                    result[identifier] = DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs)

        return result


class GVADetectionAdapter(Adapter):
    __provider__ = 'gva_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'raw_detections': BoolField(optional=True, default=False)})
        return params

    def configure(self):
        self.raw_detections = self.get_value_from_config('raw_detections')

    def process(self, raw, identifiers, frame_meta):
        results = []
        for identifier, prediction in zip(identifiers, raw):
            objects = prediction.get('objects', [])
            x_mins, y_mins, x_maxs, y_maxs, scores, labels = [], [], [], [], [], []
            for obj in objects:
                det = obj['detection']
                scores.append(det['confidence'])
                labels.append(det['label_id'])
                if not self.raw_detections:
                    x_min, y_min, x_max, y_max = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
                else:
                    bbox = det['bounding_box']
                    x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)
            results.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results


class GVAClassificationAdapter(Adapter):
    __provider__ = 'gva_classification'

    def process(self, raw, identifiers, frame_meta):
        results = []
        for identifier, image_data in zip(identifiers, raw):
            data = image_data['objects'][0]['tensors'][0]["data"]
            results.append(ClassificationPrediction(identifier, data))

        return results

""""
Copyright (c) 2018-2024 Intel Corporation

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
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..utils import read_txt
from ..representation import TextDetectionAnnotation


class RCTWPreprocessedConverter(FileBasedAnnotationConverter):
    __provider__ = 'rctw_preprocessed'

    def convert(self, check_content=False, **kwargs):
        lines = read_txt(self.annotation_file)
        annotations = []
        for line in lines:
            identifier, ann = line.strip().split('\t')
            decoded = json.loads(ann)
            all_points, transcriptions, difficult = [], [], []
            for box in decoded:
                points = box['points']
                transcript = box['transcription']
                is_difficult = transcript in ['*', '###']
                all_points.append(points)
                transcriptions.append(transcript)
                if is_difficult:
                    difficult.append(len(transcriptions) - 1)
            annotation = TextDetectionAnnotation(identifier, all_points, transcriptions)
            annotation.metadata['difficult_boxes'] = difficult
            annotations.append(annotation)
        return ConverterReturn(annotations, None, None)

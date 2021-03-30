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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import FeaturesRegressionAnnotation
from ..utils import read_csv


class TacotronDataConverter(FileBasedAnnotationConverter):
    __provider__ = 'tacotron2_data_converter'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = read_csv(self.annotation_file, delimiter='\t')
        annotations = []
        num_iter = len(examples)
        for example_id, example in enumerate(examples):
            gt = example['synthesis']
            identifier = [
                example['text_encoder_out'],
                example['domain'],
                example['f0_labels'],
                example['bert_embedding']
            ]
            annotations.append(FeaturesRegressionAnnotation(identifier, gt, is_bin=True))
            if progress_callback and example_id % progress_interval == 0:
                progress_callback(example_id * 100 / num_iter)

        return ConverterReturn(annotations, None, None)

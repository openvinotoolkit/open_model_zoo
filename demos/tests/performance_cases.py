# Copyright (c) 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
#from args import DataPatternArg

from cases import BASE, single_option_cases

THREADS_NUM = os.cpu_count()


class PerformanceParser:
    def __init__(self, demo):
        self.filename = demo.subdirectory.replace('/', '_') + '.csv'
        self.model_keys = demo.model_keys

    def __call__(self, output, test_case, device):
        result = self.parse_metrics(output)
        self.write_to_csv(result, test_case, device)

    def parse_metrics(self, output):

        def get_metric(name):
            pattern = re.compile(r'{}: (([0-9]+)\.[0-9]+)'.format(name))
            metric = pattern.search(" ".join(output.split()))
            return metric.group(1) if metric else 'N/A'

        stages_to_parse = ('Latency', 'FPS', 'Decoding', 'Preprocessing',
                           'Inference', 'Postprocessing', 'Rendering')
        return {name : get_metric(name) for name in stages_to_parse}

    def write_to_csv(self, result, test_case, device):
        result['Requests'] = test_case.options.get('-nireq', '-')
        result['Streams'] = test_case.options.get('-nstreams', '-')
        result['Threads'] = test_case.options.get('-nthreads', '-')

        if not os.path.isfile(self.filename):
            models_col = [f"Model {key}" for key in self.model_keys]
            precisions_col = [f"Precision {key}" for key in self.model_keys]
            columns = ','.join(['Device', *precisions_col, *models_col, *result.keys()])
            with open(self.filename, 'w') as f:
                print(columns, file=f)

        precisions = [test_case.options[key].precision if key in test_case.options else '-'
                      for key in self.model_keys]
        models_names = [test_case.options[key].name if key in test_case.options else '-'
                        for key in self.model_keys]
        data = ','.join([device, *precisions, *models_names, *result.values()])
        with open(self.filename, 'a') as f:
            print(data, file=f)


DEMOS = [
    BASE['interactive_face_detection_demo/cpp_gapi'].add_parser(PerformanceParser),

    BASE['interactive_face_detection_demo/cpp'].add_parser(PerformanceParser),

    BASE['object_detection_demo/python']
        .only_models(['person-detection-0200', 'yolo-v2-tf'])
    # TODO: create large -i for performance scenario
    #    .update_option({'-i': DataPatternArg('action-recognition')})
        .add_test_cases(single_option_cases('-nireq', '3', '5'),
                        single_option_cases('-nstreams', '3', '4'),
                        single_option_cases('-nthreads', str(THREADS_NUM), str(THREADS_NUM - 2)))
        .add_parser(PerformanceParser),

    BASE['bert_named_entity_recognition_demo/python']
        .update_option({'--dynamic_shape': None})
        .only_devices(['CPU'])
        .add_parser(PerformanceParser),

    BASE['gpt2_text_prediction_demo/python']
        .update_option({'--dynamic_shape': None})
        .only_devices(['CPU'])
        .add_parser(PerformanceParser),

    BASE['speech_recognition_wav2vec_demo/python']
        .update_option({'--dynamic_shape': None})
        .only_devices(['CPU'])
        .add_parser(PerformanceParser),
]

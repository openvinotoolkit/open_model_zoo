"""
 Copyright (C) 2020-2024 Intel Corporation

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

import logging as log


def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    return result

def log_latency_per_stage(*pipeline_metrics):
    stages = ('Decoding', 'Preprocessing', 'Inference', 'Postprocessing', 'Rendering')
    for stage, latency in zip(stages, pipeline_metrics):
        log.info('\t{}:\t{:.1f} ms'.format(stage, latency))

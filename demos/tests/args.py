# Copyright (c) 2019 Intel Corporation
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

import collections
import shutil

from pathlib import Path

ArgContext = collections.namedtuple('ArgContext',
    ['test_data_dir', 'dl_dir', 'model_info', 'image_sequences', 'image_sequence_dir'])

class TestDataArg:
    def __init__(self, rel_path):
        self.rel_path = rel_path

    def resolve(self, context):
        return str(context.test_data_dir / self.rel_path)

def image_net_arg(id):
    return TestDataArg('ILSVRC2012_img_val/ILSVRC2012_val_{}.JPEG'.format(id))

class ModelArg:
    def __init__(self, name, precision='FP32'):
        self.name = name
        self.precision = precision

    def resolve(self, context):
        return str(context.dl_dir / context.model_info[self.name]["subdirectory"] / self.precision / (self.name + '.xml'))

class ImagePatternArg:
    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq_dir = context.image_sequence_dir / self.sequence_name
        seq = [Path(image.resolve(context))
            for image in context.image_sequences[self.sequence_name]]

        assert len(set(image.suffix for image in seq)) == 1, "all images in the sequence must have the same extension"
        assert '%' not in seq[0].suffix

        name_format = 'input-%04d' + seq[0].suffix

        if not seq_dir.is_dir():
            seq_dir.mkdir(parents=True)

            for index, image in enumerate(context.image_sequences[self.sequence_name]):
                shutil.copyfile(image.resolve(context), str(seq_dir / (name_format % index)))

        return str(seq_dir / name_format)

class ImageDirectoryArg:
    def __init__(self, sequence_name):
        self.backend = ImagePatternArg(sequence_name)

    def resolve(self, context):
        pattern = self.backend.resolve(context)
        return str(Path(pattern).parent)

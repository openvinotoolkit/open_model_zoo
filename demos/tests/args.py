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
    ['test_data_dir', 'dl_dir', 'model_info', 'data_sequences', 'data_sequence_dir'])

RequestedModel = collections.namedtuple('RequestedModel', ['name', 'precisions'])

OMZ_DIR = Path(__file__).parents[2].resolve()


class TestDataArg:
    def __init__(self, rel_path):
        self.rel_path = rel_path

    def resolve(self, context):
        return str(context.test_data_dir / self.rel_path)


def image_net_arg(id):
    return TestDataArg('ILSVRC2012_img_val/ILSVRC2012_val_{}.JPEG'.format(id))


def brats_arg(id):
    return TestDataArg('BraTS/{}'.format(id))


def image_retrieval_arg(id):
    return TestDataArg('Image_Retrieval/{}'.format(id))


class Arg:
    @property
    def required_models(self):
        return []

    def resolve(self, context):
        raise NotImplementedError


class ModelArg(Arg):
    def __init__(self, name, precision='FP32'):
        self.name = name
        self.precision = precision

    def resolve(self, context):
        return str(context.dl_dir / context.model_info[self.name]["subdirectory"] / self.precision / (self.name + '.xml'))

    @property
    def required_models(self):
        return [RequestedModel(self.name, [self.precision])]


class ModelFileArg(Arg):
    def __init__(self, model_name, file_name):
        self.model_name = model_name
        self.file_name = file_name

    def resolve(self, context):
        return str(context.dl_dir / context.model_info[self.model_name]["subdirectory"] / self.file_name)

    @property
    def required_models(self):
        return [RequestedModel(self.model_name, [])]


class DataPatternArg(Arg):
    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq_dir = context.data_sequence_dir / self.sequence_name
        seq = [Path(data.resolve(context))
            for data in context.data_sequences[self.sequence_name]]

        assert len({data.suffix for data in seq}) == 1, "all images in the sequence must have the same extension"
        assert '%' not in seq[0].suffix

        name_format = 'input-%04d' + seq[0].suffix

        if not seq_dir.is_dir():
            seq_dir.mkdir(parents=True)

            for index, data in enumerate(context.data_sequences[self.sequence_name]):
                shutil.copyfile(data.resolve(context), str(seq_dir / (name_format % index)))

        return str(seq_dir / name_format)


class DataDirectoryArg(Arg):
    def __init__(self, sequence_name):
        self.backend = DataPatternArg(sequence_name)

    def resolve(self, context):
        pattern = self.backend.resolve(context)
        return str(Path(pattern).parent)


class DataDirectoryOrigFileNamesArg(Arg):
    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq_dir = context.data_sequence_dir / self.sequence_name
        seq = [data.resolve(context)
            for data in context.data_sequences[self.sequence_name]]

        if not seq_dir.is_dir():
            seq_dir.mkdir(parents=True)

            for seq_item in seq:
                shutil.copyfile(seq_item, str(seq_dir / Path(seq_item).name))

        return str(seq_dir)

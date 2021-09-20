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

from ..utils import read_txt, check_file_existence
from ..representation import ClassificationAnnotation
from ..data_readers import ClipIdentifier
from ..config import PathField, NumberField, StringField

from .format_converter import BaseFormatConverter, ConverterReturn


class RawFramesSegmentedRecord:
    def __init__(self, row):
        self._data = row

        assert self.video_num_frames > 0
        assert self.num_frames > 0
        assert self.fps > 0
        assert self.label >= 0
        assert self.clip_start >= self.video_start >= 0
        assert self.video_end >= self.clip_end >= 0

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

    @property
    def clip_start(self):
        return int(self._data[2])

    @property
    def clip_end(self):
        return int(self._data[3])

    @property
    def video_start(self):
        return int(self._data[4])

    @property
    def video_end(self):
        return int(self._data[5])

    @property
    def fps(self):
        return float(self._data[6])

    @property
    def num_frames(self):
        return self.clip_end - self.clip_start

    @property
    def video_num_frames(self):
        return self.video_end - self.video_start


class MSASLContiniousConverter(BaseFormatConverter):
    __provider__ = 'continuous_clip_action_recognition'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description="Path to annotation file."),
            'data_dir': PathField(is_directory=True, description="Path to data directory."),
            'out_fps': NumberField(
                value_type=float, optional=True, min_value=1, default=15, description="Output FPS."
            ),
            'clip_length': NumberField(
                value_type=int, optional=True, min_value=1, default=16, description="Clip length."
            ),
            'img_prefix': StringField(optional=True, default='img_', description="Images prefix")
        })

        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.data_dir = self.get_value_from_config('data_dir')
        self.out_fps = self.get_value_from_config('out_fps')
        self.clip_length = self.get_value_from_config('clip_length')
        self.img_prefix = self.get_value_from_config('img_prefix')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        records = self.load_annotations(self.annotation_file)

        annotations = []
        num_iterations = len(records)
        content_errors = None if not check_content else []
        for record_idx, record in enumerate(records):
            if progress_callback is not None and record_idx % progress_interval == 0:
                progress_callback(record_idx * 100 / num_iterations)

            frame_indices = self.get_indices(record, self.out_fps, self.clip_length)
            frames = ['{}{:05d}.jpg'.format(self.img_prefix, idx) for idx in frame_indices]

            identifier = ClipIdentifier(record.path, record_idx, frames)

            if check_content:
                content_errors.extend([
                    '{}: does not exist'.format(self.data_dir / frame)
                    for frame in frames if not check_file_existence(self.data_dir / frame)
                ])

            annotations.append(ClassificationAnnotation(identifier, record.label))

        return ConverterReturn(annotations, {}, content_errors)

    @staticmethod
    def load_annotations(ann_file):
        return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in read_txt(ann_file)]

    @staticmethod
    def get_indices(record, output_fps, out_clip_size):
        time_step = int(round(float(record.fps) / float(output_fps)))
        if time_step < 1:
            time_step = 1

        input_length, output_length = time_step * out_clip_size, out_clip_size

        if record.video_num_frames < input_length:
            indices = [i * time_step + 1 for i in range(record.video_num_frames // time_step)]

            num_rest = output_length - len(indices)
            if num_rest > 0:
                num_before = num_rest // 2
                num_after = num_rest - num_before
                indices = [indices[0]] * num_before + indices + [indices[-1]] * num_after
        else:
            if record.num_frames < input_length:
                shift_end = min(record.video_end - input_length + 1, record.clip_start + 1)
                start_pos = shift_end - 1
            else:
                shift_start = record.clip_start
                shift_end = record.clip_end - input_length + 1
                start_pos = (shift_start + shift_end) // 2

            indices = [start_pos + i * time_step + 1 for i in range(output_length)]

        return indices

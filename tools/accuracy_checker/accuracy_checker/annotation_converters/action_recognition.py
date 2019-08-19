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

from ..utils import read_json, read_txt, check_file_existence
from ..representation import ClassificationAnnotation
from ..data_readers import ClipIdentifier
from ..config import PathField, NumberField, StringField

from .format_converter import BaseFormatConverter, ConverterReturn


class ActionRecognitionConverter(BaseFormatConverter):
    __provider__ = 'clip_action_recognition'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description="Path to annotation file."),
            'data_dir': PathField(is_directory=True, description="Path to data directory."),
            'clips_per_video': NumberField(
                value_type=int, optional=True, min_value=0, default=3, description="Number of clips per video."
            ),
            'clip_duration': NumberField(
                value_type=int, optional=True, min_value=0, default=16, description="Clip duration."
            ),
            'temporal_stride': NumberField(
                value_type=int, optional=True, min_value=0, default=2, description="Temporal Stride."
            ),
            'subset': StringField(
                choices=['train', 'test', 'validation'], default='validation',
                optional=True, description="Subset: train, test or validation."
            )
        })

        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.data_dir = self.get_value_from_config('data_dir')
        self.clips_per_video = self.get_value_from_config('clips_per_video')
        self.clip_duration = self.get_value_from_config('clip_duration')
        self.temporal_stride = self.get_value_from_config('temporal_stride')
        self.subset = self.get_value_from_config('subset')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        full_annotation = read_json(self.annotation_file)
        label_map = dict(enumerate(full_annotation['labels']))
        video_names, annotation = self.get_video_names_and_annotations(full_annotation['database'], self.subset)
        class_to_idx = {v: k for k, v in label_map.items()}

        videos = []
        for video_name, annotation in zip(video_names, annotation):
            video_path = self.data_dir / video_name
            if not video_path.exists():
                continue

            n_frames_file = video_path / 'n_frames'
            n_frames = (
                int(read_txt(n_frames_file)[0].rstrip('\n\r')) if n_frames_file.exists()
                else len(list(video_path.glob('*.jpg')))
            )
            if n_frames <= 0:
                continue

            begin_t = 1
            end_t = n_frames
            sample = {
                'video': video_path,
                'video_name': video_name,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': video_name,
                'label': class_to_idx[annotation['label']]
            }

            videos.append(sample)

        videos = sorted(videos, key=lambda v: v['video_id'].split('/')[-1])

        clips = []
        for video in videos:
            for clip in self.get_clips(video, self.clips_per_video, self.clip_duration, self.temporal_stride):
                clips.append(clip)

        annotations = []
        num_iterations = len(clips)
        content_errors = None if not check_content else []
        for clip_idx, clip in enumerate(clips):
            if progress_callback is not None and clip_idx % progress_interval:
                progress_callback(clip_idx * 100 / num_iterations)
            identifier = ClipIdentifier(clip['video_name'], clip_idx, clip['frames'])
            if check_content:
                content_errors.extend([
                    '{}: does not exist'.format(self.data_dir / frame)
                    for frame in clip['frames'] if not check_file_existence(self.data_dir / frame)
                ])
            annotations.append(ClassificationAnnotation(identifier, clip['label']))

        return ConverterReturn(annotations, {'label_map': label_map}, content_errors)

    @staticmethod
    def get_clips(video, clips_per_video, clip_duration, temporal_stride=1):
        num_frames = video['n_frames']
        clip_duration *= temporal_stride

        if clips_per_video == 0:
            step = clip_duration
        else:
            step = max(1, (num_frames - clip_duration) // (clips_per_video - 1))

        for clip_start in range(1, 1 + clips_per_video * step, step):
            clip_end = min(clip_start + clip_duration, num_frames + 1)

            clip_idxs = list(range(clip_start, clip_end))

            if not clip_idxs:
                return

            # loop clip if it is shorter than clip_duration
            while len(clip_idxs) < clip_duration:
                clip_idxs = (clip_idxs * 2)[:clip_duration]

            clip = dict(video)
            frames_idx = clip_idxs[::temporal_stride]
            clip['frames'] = ['image_{:05d}.jpg'.format(frame_idx) for frame_idx in frames_idx]
            yield clip

    @staticmethod
    def get_video_names_and_annotations(data, subset):
        video_names = []
        annotations = []

        for key, value in data.items():
            this_subset = value['subset']
            if this_subset == subset:
                if subset == 'testing':
                    video_names.append('test/{}'.format(key))
                else:
                    label = value['annotations']['label']
                    video_names.append('{}/{}'.format(label, key))
                    annotations.append(value['annotations'])

        return video_names, annotations

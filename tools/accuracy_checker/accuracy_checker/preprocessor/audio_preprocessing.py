"""
Copyright (c) 2020 Intel Corporation

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

import numpy as np

from ..config import NumberField, ConfigError
from ..preprocessor import Preprocessor


class ResampleAudio(Preprocessor):
    __provider__ = 'resample_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'sample_rate': NumberField(value_type=int, min_value=1,
                                       description='Set new audio sample rate.'),
        })
        return parameters

    def configure(self):
        self.sample_rate = self.get_value_from_config('sample_rate')

    def process(self, image, annotation_meta=None):
        sample_rate = image.metadata.get('sample_rate')
        if sample_rate is None:
            raise RuntimeError('Operation "{}" can\'t resample audio: required original samplerate in metadata.'.
                               format(self.__provider__))

        if sample_rate == self.sample_rate:
            return image

        data = image.data
        duration = data.shape[1] / sample_rate
        resampled_data = np.zeros(shape=(data.shape[0], int(duration*self.sample_rate)), dtype=float)
        x_old = np.linspace(0, duration, data.shape[1])
        x_new = np.linspace(0, duration, resampled_data.shape[1])
        resampled_data[0] = np.interp(x_new, x_old, data[0])

        image.data = resampled_data
        image.metadata['sample_rate'] = self.sample_rate

        return image


class ClipAudio(Preprocessor):
    __provider__ = 'clip_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'duration': NumberField(
                value_type=float, min_value=0, optional=True, default=None,
                description="Length of audio clip in seconds."
            ),
            'duration_in_samples': NumberField(
                value_type=int, min_value=0, optional=True, default=None,
                description="Length of audio clip in samples."
            ),
            'max_clips': NumberField(
                value_type=int, min_value=1, default=999, optional=True,
                description="Maximum number of clips per audiofile."
            ),
            'overlap': NumberField(
                value_type=float, min_value=0.0, max_value=1.0, optional=True,
                description="Overlapping part for each clip."
            ),
            'overlap_in_samples': NumberField(
                value_type=int, min_value=0, optional=True,
                description="Overlapping part for each clip in samples."
            )
        })
        return parameters

    def configure(self):
        self.duration = self.get_value_from_config('duration')
        self.duration_in_samples = self.get_value_from_config('duration_in_samples')
        if not self.duration and not self.duration_in_samples:
            raise ConfigError("Preprocessor {}: clip duration didn\'t set.".format(self.__provider__))
        if self.duration and self.duration_in_samples:
            raise ConfigError("Preprocessor {}: clip duration is set in multiple way.".format(self.__provider__))
        self.max_clips = self.get_value_from_config('max_clips')
        self.overlap = self.get_value_from_config('overlap')
        self.overlap_in_samples = self.get_value_from_config('overlap_in_samples')
        if self.overlap and self.overlap_in_samples:
            raise ConfigError("Preprocessor {}: clip overlapping is set in multiple way.".format(self.__provider__))

    def process(self, image, annotation_meta=None):
        data = image.data
        sample_rate = image.metadata.get('sample_rate')
        if sample_rate is None:
            raise RuntimeError('Operation "{}" failed: required "sample rate" in metadata.'.
                               format(self.__provider__))
        audio_duration = data.shape[1]
        clip_duration = self.duration * sample_rate if self.duration else self.duration_in_samples
        clipped_data = []
        hop = clip_duration
        if self.overlap is not None:
            hop = int((1 - self.overlap) * hop)
        elif self.overlap_in_samples is not None:
            hop = hop - self.overlap_in_samples

        if hop > clip_duration:
            raise ConfigError("Preprocessor {}: clip overlapping exceeds clip length.".format(self.__provider__))

        for clip_no, clip_start in enumerate(range(0, audio_duration, hop)):
            if clip_start + clip_duration > audio_duration or clip_no >= self.max_clips:
                break
            clip = data[:, clip_start: clip_start + clip_duration]
            clipped_data.append(clip)

        image.data = clipped_data
        image.metadata['multi_infer'] = True

        return image


class NormalizeAudio(Preprocessor):
    __provider__ = 'audio_normalization'

    def process(self, image, annotation_meta=None):
        sound = image.data
        sound = (sound - np.mean(sound)) / (np.std(sound) + 1e-15)
        image.data = sound

        return image

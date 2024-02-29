"""
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

import pickle  # nosec B403  # disable unsafe pickle check
import numpy as np
from ..config import PathField, NumberField, BoolField, ConfigError
from ..representation import UrlClassificationAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn

class UrlClassificationConverter(BaseFormatConverter):
    __provider__ = 'urlnet'
    annotation_types = (UrlClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='Path to test dataset file'),
            'input_file': PathField(description='Path to input_1 tokens data (pickle) file'),
            'lower_input': BoolField(optional=True, default=True,
                                     description='Lower case dataset input text'),
            'num_input_tokens': NumberField(value_type=int, optional=True, default=100,
                                            description='Number input tokens'),
            'num_tokens_subsets': NumberField(value_type=int, optional=True, default=4,
                                            description='Number token subsets')
        })
        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.input_1_test = self.get_value_from_config('input_file')
        self.lower_input = self.get_value_from_config('lower_input')
        self.sampling_num = self.get_value_from_config('num_input_tokens')
        self.sub_num = self.get_value_from_config('num_tokens_subsets')

        if self.sampling_num < self.sub_num:
            raise ConfigError(
                f'num_input_tokens ({self.sampling_num}) must be higher than num_tokens_subsets ({self.sub_num})')
        if self.sampling_num % self.sub_num != 0:
            raise ConfigError(
                f'num_input_tokens ({self.sampling_num}) must be a multiple of num_tokens_subsets ({self.sub_num})')

    def texts_to_char_seq(self, texts):
        sub_sampling_len = int(self.sampling_num / self.sub_num)
        sampling_chars = []
        for text in texts:
            sampling_char = []
            char_list = [
                ord(i) if ord(i) < 128 else 0
                for i in list(text)
            ]
            if len(char_list) % self.sub_num != 0:
                char_list.extend([0] * (self.sub_num - len(char_list) % self.sub_num))

            sub_chars_len = int(len(char_list) / self.sub_num)
            if sub_chars_len >= sub_sampling_len:
                sampling_char_tmp = [
                    char_list[i: i + sub_sampling_len]
                    for i in range(0, len(char_list), sub_chars_len)
                ]
            else:
                sampling_char_tmp = [
                    char_list[i: i + sub_chars_len]
                    for i in range(0, len(char_list), sub_chars_len)
                ]
                for sub_char_seq in sampling_char_tmp:
                    sub_char_seq.extend(
                        [0] * (sub_sampling_len - len(sub_char_seq))
                    )
            for sub in sampling_char_tmp:
                sampling_char.extend(sub)
            sampling_chars.append(sampling_char)
        return np.array(sampling_chars, dtype='int32')



    def read_data(self):
        with open(self.annotation_file, 'r', encoding='utf-8') as file:
            urls = []
            labels = []
            for line in file:
                items = line.split("\t")
                if len(items) != 2:
                    continue
                if items[1].strip() == "":
                    continue
                label = int(items[0])
                labels.append(label)
                text = items[1][:-1]
                if self.lower_input:
                    text = text.lower()
                urls.append(text)
        return urls, labels

    @staticmethod
    def get_url_annotation(i, input1, input2, label):
        identifier = [
            'input_1_{}'.format(i),
            'input_2_{}'.format(i)
        ]

        return UrlClassificationAnnotation(identifier, label, np.array(input1), np.array(input2))

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        urls, labels = self.read_data()

        with open(self.input_1_test, 'rb') as f:
            input_words_tokens = pickle.load(f)  # nosec B301  # disable unsafe pickle check

        inpit_chars_tokens =  self.texts_to_char_seq(urls)

        annotations = []
        for i, label in enumerate(labels):
            input_1 = input_words_tokens[i]
            input_2 = inpit_chars_tokens[i]

            annotations.append(self.get_url_annotation(i, input_1, input_2, label))

        return ConverterReturn(annotations, None, None)

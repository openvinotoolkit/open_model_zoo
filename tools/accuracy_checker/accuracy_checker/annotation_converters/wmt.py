"""
Copyright (c) 2018-2020 Intel Corporation

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

import re
from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField
from ..representation import MachineTranslationAnnotation


def _clean(sentence, subword_option=None):
    sentence = sentence.strip()
    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
    # SPM
    if subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence


class WMTConverter(BaseFormatConverter):
    __provider__ = 'wmt'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'input_file': PathField(description='path to input file'),
                'reference_file': PathField(description='path to file with reference for translation')
            }
        )
        return configuration_parameters

    def configure(self):
        self.input_file = self.get_value_from_config('input_file')
        self.reference_file = self.get_value_from_config('reference_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        with open(str(self.input_file), 'r', encoding="utf-8") as input_f:
            input_lines = input_f.readlines()

        subword_option = self.reference_file.name.split('.')[2]
        with open(str(self.reference_file), 'r', encoding="utf-8") as ref_f:
            reference_lines = ref_f.readlines()

        reference_list = []
        for reference in reference_lines:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))

        input_list = []
        for input_line in input_lines:
            input_line = _clean(input_line)
            input_list.append(input_line.split(" "))

        annotations = []
        num_iterations = len(input_list)
        for identifier, (source, ref) in enumerate(zip(input_list, reference_list)):
            annotations.append(MachineTranslationAnnotation(identifier, source, ref))
            if progress_callback and identifier % progress_interval == 0:
                progress_callback(identifier / num_iterations * 100)

        return ConverterReturn(annotations, None, None)

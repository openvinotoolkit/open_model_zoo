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

from collections import namedtuple

import numpy as np

from ..representation import TextClassificationAnnotation, RelationClassificationAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import get_tokenizer, CLS_ID, SEP_ID


class ERNIEFewrelConverter(BaseFormatConverter):
    __provider__ = "ernie_fewrel"

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'kgentity_file': PathField(description="Path to knowledge graph entity file."),
            'vocab_file': PathField(description='Path to vocabulary file.', optional=True),
            'sentence_piece_model_file': PathField(description='sentence piece model for tokenization', optional=True),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after WordPiece tokenization.',
                optional=True, default=128, value_type=int
            ),
            'max_query_length': NumberField(
                description='The maximum number of tokens for the question.',
                optional=True, default=64, value_type=int
            ),
            'doc_stride': NumberField(
                description="When splitting up a long document into chunks, how much stride to take between chunks.",
                optional=True, default=128, value_type=int
            ),
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register')
        })

        return configuration_parameters

    def configure(self):
        self.testing_file = self.get_value_from_config('testing_file')
        self.kgentity_file = self.get_value_from_config('kgentity_file')
        self.max_seq_length = int(self.get_value_from_config('max_seq_length'))
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.doc_stride = self.get_value_from_config('doc_stride')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer = get_tokenizer(self.config, self.lower_case)
        self.support_vocab = 'vocab_file' in self.config

    @staticmethod
    def _load_examples(file):
        def _is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        lines = read_json(file)

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('test', i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
                    #print(line['text'][x[1]:x[2]].encode("utf-8"))
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                {
                    'id': guid,
                    'text_a': text_a,
                    'text_b': None,
                    'label': label,
                }
            )

        return examples

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = self._load_examples(self.testing_file)
        data = np.load(self.kgentity_file)
        input_ent = data['input_ent']
        ent_mask = data['ent_mask']

        annotations = []
        unique_id = 1000000000

        label_list = set([x['label'] for x in examples])
        label_list = sorted(label_list)
        label_map = {label: i for i, label in enumerate(label_list)}

        for (example_index, example) in enumerate(examples):
            ex_text_a = example['text_a'][0]   # text

            h, t = example['text_a'][1]        # entities
            h_name = ex_text_a[h[1]:h[2]]
            t_name = ex_text_a[t[1]:t[2]]

            if h[1] < t[1]:
                ex_text_a = ex_text_a[:h[1]] + "# " + h_name + " #" + ex_text_a[
                                                                      h[2]:t[1]] + "$ " + t_name + " $" + ex_text_a[
                                                                                                          t[2]:]
            else:
                ex_text_a = ex_text_a[:t[1]] + "$ " + t_name + " $" + ex_text_a[
                                                                      t[2]:h[1]] + "# " + h_name + " #" + ex_text_a[
                                                                                                          h[2]:]

            if h[1] < t[1]:
                h[1] += 2
                h[2] += 2  # word in h[1]:h[2] positions moves right by "#($) " or "$(#) " (2 symbols)
                t[1] += 6
                t[2] += 6  # word in t[1]:t[2] positions moves right by "#($) " * 3 (6 symbols)
            else:
                h[1] += 6
                h[2] += 6
                t[1] += 2
                t[2] += 2

            tokens_a = self.tokenizer.tokenize(ex_text_a)

            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)


            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            identifier = ['input_ids_{}'.format(example_index),
                          'input_mask_{}'.format(example_index),
                          'segment_ids_{}'.format(example_index),
                          'input_ent_{}'.format(example_index),
                          'ent_mask_{}'.format(example_index),
                          'label_ids_{}'.format(example_index),
                        ]

            annotation = RelationClassificationAnnotation(
                identifier,
                np.array(label_map[example['label']]),
                np.array(input_ids),
                np.array(input_mask),
                np.array(segment_ids),
                input_ent[example_index],
                ent_mask[example_index].reshape(self.max_seq_length),
                np.array(label_map[example['label']]),
                tokens
            )
            annotations.append(annotation)
            unique_id += 1
        return ConverterReturn(annotations, None, None)


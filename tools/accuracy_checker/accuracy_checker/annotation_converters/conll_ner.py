""""
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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ._nlp_common import WordPieceTokenizer
from ..config import BoolField, PathField, NumberField
from ..representation import BERTNamedEntityRecognitionAnnotation


class CONLLDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'conll_ner'
    label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'lower_case': BoolField(optional=True, default=False),
            'vocab_file': PathField(description='Path to vocabulary file for word piece tokenizer'),
            'max_len': NumberField(optional=True, default=128, value_type=int, description='max_sequence_length'),
            'pad_input': BoolField(
                optional=True, default=True, description='Should input be padded to max length or not'
            ),
            'include_special_token_labels': BoolField(
                optional=True, default=False, description='Should special tokens be included to labels or not'
            )
        })
        return params

    def configure(self):
        super().configure()
        self.include_spec = self.get_value_from_config('include_special_token_labels')
        self.tokenizer = WordPieceTokenizer(
            self.get_value_from_config('vocab_file'),
            lower_case=self.get_value_from_config('lower_case'), max_len=self.get_value_from_config('max_len')
        )
        if self.include_spec:
            self.label_list.extend(['[CLS]', '[SEP]'])
        self.pad = self.get_value_from_config('pad_input')

    def convert(self, check_content=False, **kwargs):
        sents, labels = self.read_annotation()
        annotations = self.convert_examples_to_features(sents, labels)
        label_map = dict(enumerate(self.label_list, int(self.include_spec)))
        return ConverterReturn(annotations, {'label_map': label_map}, None)

    def read_annotation(self):
        with self.annotation_file.open(mode='r') as lines:
            sentence, label, sentences, labels = [], [], [], []
            for line in lines:
                if not line or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        sentences.append(' '.join(sentence))
                        labels.append(label)
                        sentence = []
                        label = []
                    continue
                splits = line.split(' ')
                sentence.append(splits[0])
                label.append(splits[-1][:-1])

            if len(sentence) > 0:
                sentences.append(' '.join(sentence))
                labels.append(label)

            return sentences, labels

    def convert_examples_to_features(self, sentences, labels):
        label_map = {label: i for i, label in enumerate(self.label_list, int(self.include_spec))}

        features = []
        max_seq_length = self.tokenizer.max_len

        for ex_index, (text, label) in enumerate(zip(sentences, labels)):
            tokens = []
            labels = []
            valid = []
            label_mask = []
            for i, word in enumerate(text.split(' ')):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = label[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, int(self.include_spec))
            label_mask.insert(0, int(self.include_spec))
            label_ids.append(label_map.get('[CLS]', -1))
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(label_map[labels[i]])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(int(self.include_spec))
            label_mask.append(int(self.include_spec))
            label_ids.append(label_map.get('[SEP]', -1))
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            if self.pad:
                input_ids, input_mask, segment_ids, label_ids, valid, label_mask = pad_inputs(
                    input_ids, input_mask, segment_ids, label_ids, valid, label_mask, max_seq_length
                )

            identifier = [
                'input_ids_{}'.format(ex_index),
                'input_mask_{}'.format(ex_index),
                'segment_ids_{}'.format(ex_index),
            ]
            features.append(
                BERTNamedEntityRecognitionAnnotation(
                    identifier, input_ids, input_mask, segment_ids, label_ids, valid, label_mask
                ))

        return features


def pad_inputs(input_ids, input_mask, segment_ids, label_ids, valid, label_mask, max_seq_length):
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        valid.append(0)
        label_mask.append(0)
    while len(label_ids) < max_seq_length:
        label_ids.append(0)
        label_mask.append(0)

    return input_ids, input_mask, segment_ids, label_ids, valid, label_mask

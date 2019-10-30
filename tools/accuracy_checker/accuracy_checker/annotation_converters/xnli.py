from collections import namedtuple
import csv
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None


from ..config import PathField, StringField, NumberField, BoolField, ConfigError
from ..representation import TextClassificationAnnotation
from ..utils import string_to_list
from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import Tokenizer, truncate_seq_pair


InputExample = namedtuple('InputExample', ['guid', 'text_a', 'text_b', 'label'])

labels = ["contradiction", "entailment", "neutral"]
label_map = dict(enumerate(labels))
reversed_label_map = {value: key for key, value in label_map.items()}


class XNLIDatasetConverter(BaseFormatConverter):
    __provider__ = 'xnli'
    annotation_types = (TextClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='path to annotation file in json or tsv format'),
            'language_filter': StringField(
                description='comma-separated list of languages for selection only appropriate annotations.'
                'If not provided full dataset used',
                optional=True
                ),
            'vocab_file': PathField(description='Path to vocabulary file.'),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after WordPiece tokenization.',
                optional=True, default=128
            ),
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register')
        })

        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.language_filter = self.get_value_from_config('language_filter')
        if self.language_filter is not None:
            self.language_filter = string_to_list(self.language_filter)
        self.vocab_file = self.get_value_from_config('vocab_file')
        self.max_seq_length = self.get_value_from_config('max_seq_length')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer = Tokenizer(self.vocab_file, self.lower_case)

    def read_tsv(self):
        lines = []
        with self.annotation_file.open('r') as ann_file:
            reader = csv.reader(ann_file, delimiter="\t", quotechar=None)
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                guid = "dev-{}".format(idx)
                language = line[0]
                if self.language_filter and language not in self.language_filter:
                    continue
                label = reversed_label_map[line[1]]
                text_a = line[6]
                text_b = line[7]
                lines.append(InputExample(guid, text_a, text_b, label))

        return lines

    def convert_single_example(self, example):
        identifier = [
            'input_ids_{}'.format(example.guid),
            'input_mask_{}'.format(example.guid),
            'segment_ids_{}'.format(example.guid)
        ]
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:self.max_seq_length - 2]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_size = self.max_seq_length - len(input_ids)
        if padding_size:
            padding = [0] * padding_size
            input_ids.extend(padding)
            input_mask.extend(padding)
            segment_ids.extend(padding)

        return TextClassificationAnnotation(
            identifier, example.label, np.array(input_ids), np.array(input_mask), np.array(segment_ids), tokens
        )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = self.read_tsv()
        annotations = []
        num_iter = len(examples)
        for example_id, example in enumerate(examples):
            annotations.append(self.convert_single_example(example))
            if progress_callback and example_id % progress_interval == 0:
                progress_callback(example_id * 100 / num_iter)

        return ConverterReturn(annotations, {'label_map': label_map}, None)


class BertXNLITFRecordConverter(BaseFormatConverter):
    __provider__ = 'bert_xnli_tf_record'
    annotation_types = (TextClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='path to predict.tf_record format'),
        })

        return params

    def configure(self):
        if tf is None:
            raise ConfigError(
                'bert_xnli_tf_record converter requires Tensorflow installation. Please install it first.'
            )
        self.annotation_file = self.get_value_from_config('annotation_file')

    def read_tf_record(self):
        record_iterator = tf.python_io.tf_record_iterator(path=str(self.annotation_file))
        record_list = []
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            input_ids = example.features.feature['input_ids'].int64_list.value
            input_mask = example.features.feature['input_mask'].int64_list.value
            label_ids = example.features.feature['label_ids'].int64_list.value
            segment_ids = example.features.feature['segment_ids'].int64_list.value
            record_list.append([input_ids, input_mask, segment_ids, label_ids])
        return record_list

    @staticmethod
    def convert_single_example(example, guid):
        identifier = [
            'input_ids_{}'.format(guid),
            'input_mask_{}'.format(guid),
            'segment_ids_{}'.format(guid)
        ]

        return TextClassificationAnnotation(
            identifier, np.array(example[3]), np.array(example[0]), np.array(example[1]), np.array(example[2]), None
        )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = self.read_tf_record()
        annotations = []
        num_iter = len(examples)

        for idx, example in enumerate(examples):
            annotations.append(self.convert_single_example(example, idx))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iter)

        return ConverterReturn(annotations, None, None)

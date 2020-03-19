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
    annotation_types = (TextClassificationAnnotation, )

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
        # DocSpan = namedtuple("DocSpan", ["start", "length"])

        label_list = set([x['label'] for x in examples])
        label_list = sorted(label_list)
        label_map = {label: i for i, label in enumerate(label_list)}

        for (example_index, example) in enumerate(examples):
            # idx = unique_id + example_index
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
            # ents = ["UNK"] + entities_a + ["UNK"]
            segment_ids = [0] * len(tokens)


            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            # padding_ = [-1] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

                # add index to make identifier unique
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

    # @staticmethod
    # def _is_max_context(doc_spans, cur_span_index, position):
    #     best_score = None
    #     best_span_index = None
    #     for (span_index, doc_span) in enumerate(doc_spans):
    #         end = doc_span.start + doc_span.length - 1
    #         if position < doc_span.start:
    #             continue
    #         if position > end:
    #             continue
    #         num_left_context = position - doc_span.start
    #         num_right_context = end - position
    #         score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    #         if best_score is None or score > best_score:
    #             best_score = score
    #             best_span_index = span_index
    #
    #     return cur_span_index == best_span_index

# -------------------------------------------------
# lines from json
# each line is similar to:
#     {'label': 'P931',
#               01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
#      'text': 'Canadian Forces Base Lahr ( IATA : LHA , ICAO : EDTL , former code EDAN ) was a military operated commercial airport located in Lahr , Germany .',
#      'ents': [['Q597321', 48, 52, 0.5], ['Q7039', 21, 25, 0.5]]}
# h_name: ents[0][1]..ents[0][2] -> text[48..52] -> "EDTL"
# t_name: ents[1][1]..ents[1][2] -> text[21..25] -> "Lahr"  - t_name

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             for x in line['ents']:
#                 if x[1] == 1:
#                     x[1] = 0
#                     #print(line['text'][x[1]:x[2]].encode("utf-8"))
#             text_a = (line['text'], line['ents'])
#             label = line['label']
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#         return examples
#
# class InputExample(object):
#     """A single training/test example for simple sequence classification."""
#
#     def __init__(self, guid, text_a, text_b=None, label=None):
#         """Constructs a InputExample.
#
#         Args:
#             guid: Unique id for the example.
#             text_a: string. The untokenized text of the first sequence. For single
#             sequence tasks, only this sequence must be specified.
#             text_b: (Optional) string. The untokenized text of the second sequence.
#             Only must be specified for sequence pair tasks.
#             label: (Optional) string. The label of the example. This should be
#             specified for train and dev examples, but not for test examples.
#         """
#         self.guid = guid
#         self.text_a = text_a
#         self.text_b = text_b
#         self.label = label


# def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold):
#     """Loads a data file into a list of `InputBatch`s."""
#
#     label_list = sorted(label_list)
#     label_map = {label: i for i, label in enumerate(label_list)}
#
#     # entity2id = {}
#     # with open("kg_embed/entity2id.txt") as fin:
#     #     fin.readline()
#     #     for line in fin:
#     #         qid, eid = line.strip().split('\t')
#     #         entity2id[qid] = int(eid)
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         ex_text_a = example.text_a[0]       # text
#         h, t = example.text_a[1]            # entities
#         h_name = ex_text_a[h[1]:h[2]]
#         t_name = ex_text_a[t[1]:t[2]]
#         if h[1] < t[1]:
#             ex_text_a = ex_text_a[:h[1]] + "# " + h_name + " #" + ex_text_a[
#                                                                   h[2]:t[1]] + "$ " + t_name + " $" + ex_text_a[t[2]:]
#         else:
#             ex_text_a = ex_text_a[:t[1]] + "$ " + t_name + " $" + ex_text_a[
#                                                                   t[2]:h[1]] + "# " + h_name + " #" + ex_text_a[h[2]:]
#
#         if h[1] < t[1]:
#             h[1] += 2
#             h[2] += 2       # word in h[1]:h[2] positions moves right by "#($) " or "$(#) " (2 symbols)
#             t[1] += 6
#             t[2] += 6       # word in t[1]:t[2] positions moves right by "#($) " * 3 (6 symbols)
#         else:
#             h[1] += 6
#             h[2] += 6
#             t[1] += 2
#             t[2] += 2
#         tokens_a, entities_a = tokenizer.tokenize(ex_text_a, [h, t])
#         assert len([x for x in entities_a if x != "UNK"]) == 2
#
#         tokens_b = None
#         if example.text_b:
#             tokens_b, entities_b = tokenizer.tokenize(example.text_b[0],
#                                                       [x for x in example.text_b[1] if x[-1] > threshold])
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(tokens_a) > max_seq_length - 2:
#                 tokens_a = tokens_a[:(max_seq_length - 2)]
#                 entities_a = entities_a[:(max_seq_length - 2)]
#
#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids: 0   0   0   0  0     0 0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambigiously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.
#         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#         ents = ["UNK"] + entities_a + ["UNK"]
#         segment_ids = [0] * len(tokens)
#
#         if tokens_b:
#             tokens += tokens_b + ["[SEP]"]
#             ents += entities_b + ["UNK"]
#             segment_ids += [1] * (len(tokens_b) + 1)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         input_ent = []
#         ent_mask = []
#         for ent in ents:
#             if ent != "UNK" and ent in entity2id:
#                 input_ent.append(entity2id[ent])
#                 ent_mask.append(1)
#             else:
#                 input_ent.append(-1)
#                 ent_mask.append(0)
#         ent_mask[0] = 1
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         padding_ = [-1] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding
#         input_ent += padding_
#         ent_mask += padding
#
#         # assert len(input_ids) == max_seq_length
#         # assert len(input_mask) == max_seq_length
#         # assert len(segment_ids) == max_seq_length
#         # assert len(input_ent) == max_seq_length
#         # assert len(ent_mask) == max_seq_length
#         #
#         # label_id = label_map[example.label]
#         # if ex_index < 0:
#         #     logger.info("*** Example ***")
#         #     logger.info("guid: %s" % (example.guid))
#         #     logger.info("tokens: %s" % " ".join(
#         #         [str(x) for x in tokens]))
#         #     logger.info("ents: %s" % " ".join(
#         #         [str(x) for x in ents]))
#         #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#         #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#         #     logger.info(
#         #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#         #     logger.info("label: %s (id = %d)" % (example.label, label_id))
#
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           input_ent=input_ent,
#                           ent_mask=ent_mask,
#                           label_id=label_id))
#     return features
#
# # Out is ndarray of shape [8,80], logits output
# # labels is a array of shape [8,],
#
# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels), outputs

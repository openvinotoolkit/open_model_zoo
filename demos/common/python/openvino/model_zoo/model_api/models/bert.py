"""
 Copyright (c) 2021 Intel Corporation
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

from .model import Model, WrapperError
from .types import DictValue, NumericalValue, StringValue, BooleanValue


class Bert(Model):
    __model__ = 'bert'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.token_cls = [self.vocab['[CLS]']]
        self.token_sep = [self.vocab['[SEP]']]
        self.token_pad = [self.vocab['[PAD]']]
        self.input_names = [i.strip() for i in self.input_names.split(',')]
        if self.inputs.keys() != set(self.input_names):
            raise WrapperError(self.__model__, 'The Wrapper expects input names: {}, actual network input names: {}'.format(
                self.input_names, list(self.inputs.keys())))
        self.max_length = self.inputs[self.input_names[0]].shape[1]

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'vocab': DictValue(),
            'input_names': StringValue(description='Comma-separated names of input layers'),
            'enable_padding': BooleanValue(
                description='Should be input sequence padded to max sequence len or not', default_value=True
            )
        })
        return parameters

    def preprocess(self, inputs):
        input_ids, attention_mask, token_type_ids = self.form_request(inputs)
        pad_len = self.pad_input(input_ids, attention_mask, token_type_ids) if self.enable_padding else 0
        meta = {'pad_len': pad_len, 'inputs': inputs}

        return self.create_input_dict(input_ids, attention_mask, token_type_ids), meta

    def form_request(self, inputs):
        raise NotImplementedError

    def pad_input(self, input_ids, attention_mask, token_type_ids):
        pad_len = self.max_length - len(input_ids)
        if pad_len < 0:
            raise ValueError("The input request is longer than max number of tokens ({})"
                             " processed by model".format(self.max_length))
        input_ids += self.token_pad * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len
        return pad_len

    def create_input_dict(self, input_ids, attention_mask, token_type_ids):
        inputs = {
            self.input_names[0]: np.array([input_ids], dtype=np.int32),
            self.input_names[1]: np.array([attention_mask], dtype=np.int32),
            self.input_names[2]: np.array([token_type_ids], dtype=np.int32),
        }
        if len(self.input_names) > 3:
            inputs[self.input_names[3]] = np.arange(len(input_ids), dtype=np.int32)[None, :]

        return inputs

    def reshape(self, new_length):
        new_shapes = {}
        for input_name, input_info in self.inputs.items():
            new_shapes[input_name] = [1, new_length]
        default_input_shape = input_info.shape
        super().reshape(new_shapes)
        self.logger.debug("\tReshape model from {} to {}".format(default_input_shape, new_shapes[input_name]))
        self.max_length = new_length if not isinstance(new_length, tuple) else new_length[1]


class BertNamedEntityRecognition(Bert):
    __model__ = 'bert-named-entity-recognition'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)

        self.output_names = list(self.outputs)
        self._check_io_number(-1, 1)

    def form_request(self, inputs):
        c_tokens_id = inputs
        input_ids = self.token_cls + c_tokens_id + self.token_sep
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        return input_ids, attention_mask, token_type_ids

    def postprocess(self, outputs, meta):
        output = outputs[self.output_names[0]]
        output = np.exp(output[0])
        score = output / output.sum(axis=-1, keepdims=True)
        labels_id = score.argmax(-1)

        filtered_labels_id = [
            (i, label_i) for i, label_i in enumerate(labels_id)
            if label_i != 0 and 0 < i < self.max_length - meta['pad_len'] - 1
        ]
        return score, filtered_labels_id


class BertEmbedding(Bert):
    __model__ = 'bert-embedding'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)

        self.output_names = list(self.outputs)
        self._check_io_number(-1, 1)

    def form_request(self, inputs):
        tokens_id, self.max_length = inputs
        input_ids = self.token_cls + tokens_id + self.token_sep
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        return input_ids, attention_mask, token_type_ids

    def postprocess(self, outputs, meta):
        output = outputs[self.output_names[0]]
        return output.squeeze(0)


class BertQuestionAnswering(Bert):
    __model__ = 'bert-question-answering'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)

        self.output_names = [o.strip() for o in self.output_names.split(',')]
        if self.outputs.keys() != set(self.output_names):
            raise WrapperError(self.__model__, 'The Wrapper output names: {}, actual network output names: {}'.format(
                self.output_names, list(self.outputs.keys())))

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'output_names': StringValue(description='Comma-separated names of output layers'),
            'max_answer_token_num': NumericalValue(value_type=int),
            'squad_ver': StringValue(),
        })
        return parameters

    def form_request(self, inputs):
        c_data, q_tokens_id = inputs
        input_ids = self.token_cls + q_tokens_id + self.token_sep + c_data.c_tokens_id + self.token_sep
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(q_tokens_id) + 2) + [1] * (len(c_data.c_tokens_id) + 1)
        return input_ids, attention_mask, token_type_ids

    def postprocess(self, outputs, meta):

        def get_score(blob_name):
            out = np.exp(outputs[blob_name].reshape((self.max_length,)))
            return out / out.sum(axis=-1)

        pad_len, (c_data, q_tokens_id) = meta['pad_len'], meta['inputs']
        # get start-end scores for context
        score_s = get_score(self.output_names[0])
        score_e = get_score(self.output_names[1])

        # index of first context token in tensor
        c_s_idx = len(q_tokens_id) + 2
        # index of last+1 context token in tensor
        c_e_idx = self.max_length - (pad_len + 1)

        # find product of all start-end combinations to find the best one
        max_score, max_s, max_e = self.find_best_answer_window(score_s, score_e, c_s_idx, c_e_idx)

        # convert to context text start-end index
        max_s = c_data.c_tokens_se[max_s][0]
        max_e = c_data.c_tokens_se[max_e][1]

        return max_score, max_s, max_e

    def find_best_answer_window(self, start_score, end_score, context_start_idx, context_end_idx):
        # get 'no-answer' score (not valid if model has been fine-tuned on squad1.x)
        score_na = 0 if '1.' in self.squad_ver else start_score[0] * end_score[0]

        context_len = context_end_idx - context_start_idx
        score_mat = np.matmul(
            start_score[context_start_idx:context_end_idx].reshape((context_len, 1)),
            end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
        )
        # reset candidates with end before start
        score_mat = np.triu(score_mat)
        # reset long candidates (>max_answer_token_num)
        score_mat = np.tril(score_mat, self.max_answer_token_num - 1)
        # find the best start-end pair
        max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
        max_score = score_mat[max_s, max_e] * (1 - score_na)

        return max_score, max_s, max_e

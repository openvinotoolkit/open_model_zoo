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
import unicodedata

from ..representation import QuestionAnsweringAnnotation
from ..utils import read_json
from ..config import PathField, NumberField

from .format_converter import BaseFormatConverter

class SQUADConverter(BaseFormatConverter):
    __provider__ = "squad"
    annotation_types = (QuestionAnsweringAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'vocab_file': PathField(description='Path to vocabulary file.'),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after WordPiece tokenization.',
                optional=True, default=128
            ),
            'max_query_length': NumberField(
                description='The maximum number of tokens for the question.',
                optional=True, default=64
            ),
            'doc_stride': NumberField(
                description="When splitting up a long document into chunks, how much stride to take between chunks.",
                optional=True, default=128
            )
        })

        return parameters

    def configure(self):
        self.testing_file = self.get_value_from_config('testing_file')
        self.vocab_file = self.get_value_from_config('vocab_file')
        self.max_seq_length = self.get_value_from_config('max_seq_length')
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.doc_stride = self.get_value_from_config('doc_stride')

    @staticmethod
    def _load_examples(file):
        def _is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        answers = []
        data = read_json(file)['data']

        for entry in data:
            for paragraph in entry['paragraphs']:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    orig_answer_text = qa["answers"]
                    is_impossible = False

                    example = {
                        'id': qas_id,
                        'question_text': question_text,
                        'tokens': doc_tokens,
                        'is_impossible': is_impossible
                    }
                    examples.append(example)
                    answers.append(orig_answer_text)
        return examples, answers

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples, answers = self._load_examples(self.testing_file)
        annotations = []
        tokenizer = Tokenizer(self.vocab_file)
        unique_id = 1000000000
        DocSpan = namedtuple("DocSpan", ["start", "length"])

        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example['question_text'])
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[:self.max_query_length]
            all_doc_tokens = []
            for (i, token) in enumerate(example['tokens']):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_doc_tokens.append(sub_token)
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(DocSpan(start_offset, length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for idx, doc_span in enumerate(doc_spans):
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                # add index to make identifier unique
                identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'segment_ids_{}'.format(idx)]
                annotation = QuestionAnsweringAnnotation(
                    identifier,
                    unique_id,
                    input_ids,
                    input_mask,
                    segment_ids,
                    tokens,
                    answers[example_index],
                )
                annotations.append(annotation)
                unique_id += 1
        return annotations

    @staticmethod
    def _is_max_context(doc_spans, cur_span_index, position):
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


class Tokenizer:
    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)

    @staticmethod
    def _run_strip_accents(text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text):
        def _is_punctuation(char):
            punct = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            if char in punct:
                return True
            cat = unicodedata.category(char)
            if cat.startswith("P"):
                return True
            return False

        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def basic_tokenizer(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")

        text = text.strip()
        tokens = text.split() if text else []
        split_tokens = []
        for token in tokens:
            token = token.lower()
            token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = " ".join(split_tokens)
        output_tokens = output_tokens.strip()
        output_tokens = output_tokens.split() if output_tokens else []
        return output_tokens

    def wordpiece_tokenizer(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")

        output_tokens = []
        text = text.strip()
        tokens = text.split() if text else []
        for token in tokens:
            chars = list(token)
            if len(chars) > 200:
                output_tokens.append("[UNK]")
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append("[UNK]")
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def tokenize(self, text):
        tokens = []
        for token in self.basic_tokenizer(text):
            for sub_token in self.wordpiece_tokenizer(token):
                tokens.append(sub_token)

        return tokens

    def convert_tokens_to_ids(self, items):
        output = []
        for item in items:
            output.append(self.vocab[item])
        return output


def load_vocab(file):
    vocab = {}
    index = 0
    with open(str(file), 'r') as reader:
        while True:
            token = reader.readline()
            if isinstance(token, bytes):
                token = token.decode("utf-8", "ignore")
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

"""
Copyright (c)  2018-2021 Intel Corporation

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
import numpy as np

from ..representation import QuestionAnsweringBiDAFAnnotation
from ..utils import read_json, UnsupportedPackage
from ..config import PathField
from .squad import SquadExample
from .format_converter import BaseFormatConverter, ConverterReturn

try:
    import nltk
except ImportError as import_error:
    nltk = UnsupportedPackage("nltk", import_error.msg)

class SQUADConverterBiDAF(BaseFormatConverter):
    __provider__ = "squad_bidaf"
    annotation_types = (QuestionAnsweringBiDAFAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
        })

        return configuration_parameters

    def configure(self):
        if isinstance(nltk, UnsupportedPackage):
            nltk.raise_error(self.__provider__)
        else:
            nltk.download('punkt')
        self.testing_file = self.get_value_from_config('testing_file')

    @staticmethod
    def _load_examples(file):
        def preprocess(s):
            return s.replace("''", '" ').replace("``", '" ')

        examples = []
        data = read_json(file)['data']

        for entry in data:
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = preprocess(paragraph["context"])
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = preprocess(qa["question"])
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        answers = qa["answers"]
                        for answer in answers:
                            answer.update({
                                "text": preprocess(answer["text"])
                            })

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples

    @staticmethod
    def _preprocess_text(text, is_context=False):
        def tokenize(s, is_context=False):
            nltk_tokens = [t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(s)]
            additional_separators = (
                "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018",
                "\u00B0")
            if is_context:
                tokens = []
                for token in nltk_tokens:
                    tokens.extend(re.split("([{}])".format("".join(additional_separators)), token))
            else:
                tokens = nltk_tokens
            assert not any(t == '<NULL>' for t in tokens)
            assert not any(' ' in t for t in tokens)
            assert not any('\t' in t for t in tokens)
            return tokens

        words = tokenize(text, is_context)
        chars = [list(w)[:16] for w in words]
        words = np.asarray([w.lower() for w in words]).reshape(-1, 1)
        chars = np.asarray([cs + [''] * (16 - len(cs)) for cs in chars]).reshape(-1, 1, 1, 16)
        return words, chars

    @staticmethod
    def _get_tokens_indexes_in_context(context, words):
        indexes = []
        rem = context.lower()
        offset = 0
        for w in words:
            idx = rem.find(w)
            assert idx >= 0
            indexes.append(idx + offset)
            offset += idx + len(w)
            rem = rem[idx + len(w):]
        return indexes

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = self._load_examples(self.testing_file)

        annotations = []
        for example_index, example in enumerate(examples):
            cw, cc = self._preprocess_text(example.context_text, True)
            qw, qc = self._preprocess_text(example.question_text)
            cw_idx_in_context = self._get_tokens_indexes_in_context(example.context_text, cw.reshape(-1))
            identifier = ['context_word_{}'.format(example_index),
                          'context_char_{}'.format(example_index),
                          'query_word_{}'.format(example_index),
                          'query_char_{}'.format(example_index)]
            annotation = QuestionAnsweringBiDAFAnnotation(
                identifier=identifier,
                title=example.title,
                context=example.context_text,
                query=example.question_text,
                answers=example.answers,
                context_word=cw,
                context_char=cc,
                query_word=qw,
                query_char=qc,
                question_id=example.qas_id,
                words_idx_in_context=cw_idx_in_context
            )
            annotations.append(annotation)

        return ConverterReturn(annotations, None, None)

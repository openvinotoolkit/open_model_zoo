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

from pathlib import Path
import re
import json
import string
import inflect
from unidecode import unidecode

import numpy as np
from ..representation import CharacterRecognitionAnnotation
from .format_converter import DirectoryBasedAnnotationConverter
from .format_converter import ConverterReturn
from ..config import PathField, NumberField


class CharParser:
    """Functor for parsing raw strings into list of int tokens.

    Examples:
        >>> parser = CharParser(['a', 'b', 'c'])
        >>> parser('abc')
        [0, 1, 2]
    """

    def __init__(
        self,
        labels,
        *,
        unk_id: int = -1,
        blank_id: int = -1,
        do_normalize: bool = True,
        do_lowercase: bool = True
    ):
        """Creates simple mapping char parser.

        Args:
            labels: List of labels to allocate indexes for. Essentially,
                this is a id to str mapping.
            unk_id: Index to choose for OOV words (default: -1).
            blank_id: Index to filter out from final list of tokens
                (default: -1).
            do_normalize: True if apply normalization step before tokenizing
                (default: True).
            do_lowercase: True if apply lowercasing at normalizing step
                (default: True).
        """

        self._labels = labels
        self._unk_id = unk_id
        self._blank_id = blank_id
        self._do_normalize = do_normalize
        self._do_lowercase = do_lowercase

        self._labels_map = {label: index for index, label in enumerate(labels)}
        self._special_labels = set([label for label in labels if len(label) > 1])

    def __call__(self, text: str):
        if self._do_normalize:
            text = self._normalize(text)
            if text is None:
                return None

        text_tokens = self._tokenize(text)
        text = ''.join(self._labels[token] for token in text_tokens)

        return text_tokens, text

    def _normalize(self, text: str) :
        text = text.strip()

        if self._do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text: str):
        tokens = []
        # Split by word for find special labels.
        for word_id, word in enumerate(text.split(' ')):
            if word_id != 0:  # Not first word - so we insert space before.
                tokens.append(self._labels_map.get(' ', self._unk_id))

            if word in self._special_labels:
                tokens.append(self._labels_map[word])
                continue

            for char in word:
                tokens.append(self._labels_map.get(char, self._unk_id))

        # If unk_id == blank_id, OOV tokens are removed.
        tokens = [token for token in tokens if token != self._blank_id]

        return tokens


class ENCharParser(CharParser):
    """Incorporates english-specific parsing logic."""

    PUNCTUATION_TO_REPLACE = {'+': 'plus', '&': 'and', '%': 'percent'}

    def __init__(self, *args, **kwargs):
        """Creates english-specific mapping char parser.

        This class overrides normalizing implementation.

        Args:
            *args: Positional args to pass to `CharParser` constructor.
            **kwargs: Key-value args to pass to `CharParser` constructor.
        """

        super().__init__(*args, **kwargs)

        self._table = self.__make_trans_table()

    def __make_trans_table(self):
        punctuation = string.punctuation

        for char in self.PUNCTUATION_TO_REPLACE:
            punctuation = punctuation.replace(char, '')

        for label in self._labels:
            punctuation = punctuation.replace(label, '')

        table = str.maketrans(punctuation, ' ' * len(punctuation))

        return table

    def _normalize(self, text: str):
        # noinspection PyBroadException
        try:
            text = clean_text(
                string=text, table=self._table, punctuation_to_replace=self.PUNCTUATION_TO_REPLACE,
            )
        except Exception:
            return None

        return text

class LibrispeechConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'librispeech'
    annotation_types = (CharacterRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'annotation_file': PathField(), 'top_n': NumberField(optional=True, default=100, value_type=int)})
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.annotation_list = []
        with self.annotation_file.open() as json_file:
            for line in json_file:
                self.annotation_list.append(json.loads(line))
        durations = [float(ann['duration']) for ann in self.annotation_list]
        sorted_by_duration = np.argsort(durations)
        self.top_n = self.get_value_from_config('top_n')
        if self.top_n is not None:
            subset = sorted_by_duration[:self.top_n]
            annotation_list = [ann for idx, ann in enumerate(self.annotation_list) if idx in subset]
            subset_file = Path('top_{}_{}'.format(self.top_n, self.annotation_file.name))
            with subset_file.open('w') as out_file:
                jsonify_content = [json.dumps(ann)+'\n' for ann in annotation_list]
                out_file.writelines(jsonify_content)
            self.annotation_list = annotation_list
        self.file_paths = [Path(ann["audio_filepath"]).name for ann in self.annotation_list]
        self.text = dict(zip(self.file_paths, [ann['text'] for ann in self.annotation_list]))
        kwargs = {'unk_id': -1, 'blank_id': -1, 'do_normalize': True}
        self.parser = ENCharParser(
            labels=[' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"],
            **kwargs
        )

    def convert(self, check_content=False, **kwargs):

        pattern = re.compile(r'([0-9\-]+)\s+(.+)')
        annotations = []
        data_folder = Path(self.data_dir)
        txts = list(data_folder.glob('**/*.txt'))
        for txt in txts:
            content = txt.open().readlines()
            for line in content:
                res = pattern.search(line)
                if res:
                    name = res.group(1)
                    transcript = res.group(2)
                    fname = txt.parent / name
                    fname = fname.with_suffix('.wav')
                    if fname.name in self.file_paths:
                        _, transcript =self.parser(self.text[fname.name])
                        annotations.append(CharacterRecognitionAnnotation(str(fname.relative_to(data_folder)), transcript.upper()))

        return ConverterReturn(annotations, None, None)

NUM_CHECK = re.compile(r'([$]?)(^|\s)(\S*[0-9]\S*)(?=(\s|$)((\S*)(\s|$))?)')

TIME_CHECK = re.compile(r'([0-9]{1,2}):([0-9]{2})(am|pm)?')
CURRENCY_CHECK = re.compile(r'\$')
ORD_CHECK = re.compile(r'([0-9]+)(st|nd|rd|th)')
THREE_CHECK = re.compile(r'([0-9]{3})([.,][0-9]{1,2})?([!.?])?$')
DECIMAL_CHECK = re.compile(r'([.,][0-9]{1,2})$')

ABBREVIATIONS_COMMON = [
    (re.compile('\\b%s\\.' % x[0]), x[1])
    for x in [
        ("ms", "miss"),
        ("mrs", "misess"),
        ("mr", "mister"),
        ("messrs", "messeurs"),
        ("dr", "doctor"),
        ("drs", "doctors"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("sr", "senior"),
        ("rev", "reverend"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("maj", "major"),
        ("col", "colonel"),
        ("lt", "lieutenant"),
        ("gen", "general"),
        ("prof", "professor"),
        ("lb", "pounds"),
        ("rep", "representative"),
        ("st", "street"),
        ("ave", "avenue"),
        ("etc", "et cetera"),
        ("jan", "january"),
        ("feb", "february"),
        ("mar", "march"),
        ("apr", "april"),
        ("jun", "june"),
        ("jul", "july"),
        ("aug", "august"),
        ("sep", "september"),
        ("oct", "october"),
        ("nov", "november"),
        ("dec", "december"),
    ]
]

ABBREVIATIONS_EXPANDED = [
    (re.compile('\\b%s\\.' % x[0]), x[1])
    for x in [
        ("ltd", "limited"),
        ("fig", "figure"),
        ("figs", "figures"),
        ("gent", "gentlemen"),
        ("ft", "fort"),
        ("esq", "esquire"),
        ("prep", "preperation"),
        ("bros", "brothers"),
        ("ind", "independent"),
        ("mme", "madame"),
        ("pro", "professional"),
        ("vs", "versus"),
        ("inc", "include"),
    ]
]

inflect = inflect.engine()


def clean_text(string, table, punctuation_to_replace):
    warn_common_chars(string)
    string = unidecode(string)
    string = string.lower()
    string = re.sub(r'\s+', " ", string)
    string = clean_numbers(string)
    string = clean_abbreviations(string)
    string = clean_punctuations(string, table, punctuation_to_replace)
    string = re.sub(r'\s+', " ", string).strip()
    return string


def warn_common_chars(string):
    pass
    # if re.search(r'[£€]', string):
    #     logging.warning("Your transcript contains one of '£' or '€' which we do not currently handle")


def clean_numbers(string):
    cleaner = NumberCleaner()
    string = NUM_CHECK.sub(cleaner.clean, string)
    return string


def clean_abbreviations(string, expanded=False):
    for regex, replacement in ABBREVIATIONS_COMMON:
        string = re.sub(regex, replacement, string)
    if expanded:
        for regex, replacement in ABBREVIATIONS_EXPANDED:
            string = re.sub(regex, replacement, string)
    return string


def clean_punctuations(string, table, punctuation_to_replace):
    for punc, replacement in punctuation_to_replace.items():
        string = re.sub('\\{}'.format(punc), " {} ".format(replacement), string)
    string = string.translate(table)
    return string


class NumberCleaner:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.curr_num = []
        self.currency = None

    def format_final_number(self, whole_num, decimal):
        if self.currency:
            return_string = inflect.number_to_words(whole_num)
            return_string += " dollar" if whole_num == 1 else " dollars"
            if decimal:
                return_string += " and " + inflect.number_to_words(decimal)
                return_string += " cent" if whole_num == decimal else " cents"
            self.reset()
            return return_string

        self.reset()
        if decimal:
            whole_num += "." + decimal
            return inflect.number_to_words(whole_num)
        else:
            # Check if there are non-numbers
            def convert_to_word(match):
                return " " + inflect.number_to_words(match.group(0)) + " "

            return re.sub(r'[0-9,]+', convert_to_word, whole_num)

    def clean(self, match):
        ws = match.group(2)
        number = match.group(3)
        _proceeding_symbol = match.group(7)

        time_match = TIME_CHECK.match(number)
        if time_match:
            string = ws + inflect.number_to_words(time_match.group(1)) + "{}{}"
            mins = int(time_match.group(2))
            min_string = ""
            if mins != 0:
                min_string = " " + inflect.number_to_words(time_match.group(2))
            ampm_string = ""
            if time_match.group(3):
                ampm_string = " " + time_match.group(3)
            return string.format(min_string, ampm_string)

        ord_match = ORD_CHECK.match(number)
        if ORD_CHECK.match(number):
            return ws + inflect.number_to_words(ord_match.group(0))

        if self.currency is None:
            # Check if it is a currency
            self.currency = match.group(1) or CURRENCY_CHECK.match(number)

        # Check to see if next symbol is a number
        # If it is a number and it has 3 digits, then it is probably a
        # continuation
        three_match = THREE_CHECK.match(match.group(6))
        if three_match:
            self.curr_num.append(number)
            return " "
        # Else we can output
        else:
            # Check for decimals
            whole_num = "".join(self.curr_num) + number
            decimal = None
            decimal_match = DECIMAL_CHECK.search(whole_num)
            if decimal_match:
                decimal = decimal_match.group(1)[1:]
                whole_num = whole_num[: -len(decimal) - 1]
            whole_num = re.sub(r'\.', '', whole_num)
            return ws + self.format_final_number(whole_num, decimal)


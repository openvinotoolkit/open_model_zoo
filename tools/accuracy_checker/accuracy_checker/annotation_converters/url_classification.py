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

import numpy as np
import tensorflow as tf
from collections import namedtuple
import numpy as np
import pickle
from bisect import bisect_left 
from keras.preprocessing.sequence import pad_sequences

from ..config import PathField, NumberField
from ..representation import UrlClassificationAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn


def is_in(a,x): 
    i = bisect_left(a,x)
    if i != len(a) and a[i] == x: 
        return True 
    else:
        return False 


def tokenize_urls_with_dict(urls, word_dict = None): 
    tokenized_urls = [] 
    word_vocab = sorted(list(word_dict.keys()))
    for url in urls:
        url_tokens = [] 
        words = url.split(' ')
        for word in words:
            if is_in(word_vocab, word): 
                word_id = word_dict[word]
            else: 
                word_id = word_dict["<UNK>"] 
            url_tokens.append(word_id)
        tokenized_urls.append(url_tokens)
    
    return tokenized_urls 



UrlInputExample = namedtuple('UrlInputExample', ['url', 'label'])
class UrlClassificationConverter(BaseFormatConverter):
    __provider__ = 'urlnet'
    annotation_types = (UrlClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='Path to urlnet dataset file'),
            'input_file': PathField(description='Path to words_dict.p pickle file'),
            'max_len_words': NumberField(
                description='The maximum number of words in a URL.',
                optional=True, default=200, value_type=int
            ),
        })
        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.words_dict = self.get_value_from_config('input_file')
        self.max_len_words = self.get_value_from_config('max_len_words')

    def read_data(self): 
        with open(self.annotation_file, 'r') as file: 
            urls = []
            labels = []
            for line in file.readlines(): 
                items = line.split('\t') 
                label = int(items[0]) 
                if label == 1: 
                    labels.append(1) 
                else: 
                    labels.append(0) 
                url = items[1][:-1]
                urls.append(url) 
        return urls, labels 


    @staticmethod
    def get_url_annotation(id, url_data, label):
        identifier = [
            'input_x_word_{}'.format(id),
            'dropout_keep_prob_{}'.format(id)
        ]

        dropout = np.float32(1)
        return UrlClassificationAnnotation(identifier, label, np.array(url_data), np.array(dropout))

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        urls, labels = self.read_data()  
        with open(self.words_dict, 'rb') as f:
            word_vocab = pickle.load(f)

        special_tokens = {}
        for key, value in word_vocab.items():
            if len(key) == 1 and not key[0].isalnum():
                special_tokens[key] = value

        for key, value in special_tokens.items():
            word_vocab['<<'+key+'>>'] = word_vocab[key]
            del word_vocab[key]

        for key, value in special_tokens.items():
            new_key = ' <<' + key + '>> '
            for i, url in enumerate(urls):
                if key in url:
                   urls[i] = url.replace(key, new_key) 

        tokenized_urls = tokenize_urls_with_dict(urls, word_vocab) 
        padded_sequences = pad_sequences(tokenized_urls, maxlen=self.max_len_words, padding='post', truncating='post')

        padded_sequences = padded_sequences.astype(np.int32)            

        annotations = []
        for id, input_x_word in enumerate(padded_sequences):
            annotations.append(self.get_url_annotation(id, input_x_word, labels[id]))

        return ConverterReturn(annotations, None, None)

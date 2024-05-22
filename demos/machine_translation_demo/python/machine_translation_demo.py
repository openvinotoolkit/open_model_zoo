#!/usr/bin/env python3

"""
 Copyright (c) 2020-2024 Intel Corporation
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
import argparse
import itertools
import logging as log
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
from openvino import Core, get_version
from tokenizers import SentencePieceBPETokenizer

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


class Translator:
    """ Language translation.

    Arguments:
        model_path (str): path to model's .xml file.
        tokenizer_src (str): path to src tokenizer.
        tokenizer_tgt (str): path to tgt tokenizer.
    """
    def __init__(self, model_path, device, tokenizer_src, tokenizer_tgt, output_name):
        self.engine = TranslationEngine(model_path, device, output_name)
        self.max_tokens = self.engine.get_max_tokens()
        self.tokenizer_src = Tokenizer(tokenizer_src, self.max_tokens)
        log.debug('Loaded src tokenizer, max tokens: {}'.format(self.max_tokens))
        self.tokenizer_tgt = Tokenizer(tokenizer_tgt, self.max_tokens)
        log.debug('Loaded tgt tokenizer, max tokens: {}'.format(self.max_tokens))

    def __call__(self, sentence, remove_repeats=True):
        """ Main translation method.

        Arguments:
            sentence (str): sentence for translate.
            remove_repeats (bool): remove repeated words.

        Returns:
            translation (str): translated sentence.
        """
        tokens = self.tokenizer_src.encode(sentence)
        assert len(tokens) == self.max_tokens, "the input sentence is too long."
        tokens = np.array(tokens).reshape(1, -1)
        translation = self.engine(tokens)
        translation = self.tokenizer_tgt.decode(translation[0], remove_repeats)
        return translation


class TranslationEngine:
    """ OpenVINO engine for machine translation.

    Arguments:
        model_path (str): path to model's .xml file.
        output_name (str): name of output blob of model.
    """
    def __init__(self, model_path, device, output_name):
        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        log.info('Reading model {}'.format(model_path))
        self.model = core.read_model(model_path)
        compiled_model = core.compile_model(self.model, args.device)
        self.infer_request = compiled_model.create_infer_request()
        log.info('The model {} is loaded to {}'.format(model_path, device))
        self.input_tensor_name = "tokens"
        self.output_tensor_name = output_name
        self.model.output(self.output_tensor_name) # ensure a tensor with the name exists

    def get_max_tokens(self):
        """ Get maximum number of tokens that supported by model.

        Returns:
            max_tokens (int): maximum number of tokens;
        """
        return self.model.input(self.input_tensor_name).shape[1]

    def __call__(self, tokens):
        """ Inference method.

        Arguments:
            tokens (np.array): input sentence in tokenized format.

        Returns:
            translation (np.array): translated sentence in tokenized format.
        """
        self.infer_request.infer({self.input_tensor_name: tokens})
        return self.infer_request.get_tensor(self.output_tensor_name).data[:]


class Tokenizer:
    """ Sentence tokenizer.

    Arguments:
        path (str): path to tokenizer's model folder.
        max_tokens (int): max tokens.
    """
    def __init__(self, path, max_tokens):
        self.tokenizer = SentencePieceBPETokenizer.from_file(
            str(path / "vocab.json"),
            str(path / "merges.txt"),
        )
        self.max_tokens = max_tokens
        self.idx = {}
        for s in ['</s>', '<s>', '<pad>']:
            self.idx[s] = self.tokenizer.token_to_id(s)

    def encode(self, sentence):
        """ Encode method for sentence.

        Arguments:
            sentence (str): sentence.

        Returns:
            tokens (np.array): encoded sentence in tokneized format.
        """
        tokens = self.tokenizer.encode(sentence).ids
        return self._extend_tokens(tokens)

    def decode(self, tokens, remove_repeats=True):
        """ Decode method for tokens.

        Arguments:
            tokens (np.array): sentence in tokenized format.
            remove_repeats (bool): remove repeated words.

        Returns:
            sentence (str): output sentence.
        """
        sentence = self.tokenizer.decode(tokens)
        for s in self.idx.keys():
            sentence = sentence.replace(s, '')
        if remove_repeats:
            sentence = self._remove_repeats(sentence)
        return sentence.lstrip()

    def _extend_tokens(self, tokens):
        """ Extend tokens.

        Arguments:
            tokens (np.array): sentence in tokenized format.

        Returns:
            tokens (np.array): extended tokens.
        """
        tokens = [self.idx['<s>']] + tokens + [self.idx['</s>']]
        pad_length = self.max_tokens - len(tokens)
        if pad_length > 0:
            tokens = tokens + [self.idx['<pad>']] * pad_length
        return tokens

    def _remove_repeats(self, sentence):
        """ Remove repeated words.

        Arguments:
            sentence (str): sentence.

        Returns:
            sentence (str): sentence in lowercase without repeated words.
        """
        tokens = sentence.lower().split()
        return " ".join(key for key, _ in itertools.groupby(tokens))


def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=Path,
                        help="Required. Path to an .xml file with a trained model.")
    parser.add_argument('--tokenizer-src', type=Path, required=True,
                        help='Required. Path to the folder with src tokenizer that contains vocab.json and merges.txt.')
    parser.add_argument('--tokenizer-tgt', type=Path, required=True,
                        help='Required. Path to the folder with tgt tokenizer that contains vocab.json and merges.txt.')
    parser.add_argument('-i', '--input', type=str, required=False, nargs='*',
                        help='Optional. Text for translation or path to the input .txt file. Replaces console input.')
    parser.add_argument('-d', '--device', default='CPU', type=str,
                        help='Optional. Specify the target device to infer on; CPU or GPU is '
                             'acceptable. The demo will look for a suitable plugin for device specified. '
                             'Default value is CPU.')
    parser.add_argument('-o', '--output', required=False, type=str,
                         help='Optional. Path to the output .txt file.')
    parser.add_argument('--output-name', type=str, default='pred',
                        help='Optional. Name of the models output node.')
    return parser

def parse_input(input):
    if not input:
        return
    sentences = []
    for text in input:
        if text.endswith('.txt'):
            try:
                with open(text, 'r', encoding='utf8') as f:
                    sentences += f.readlines()
                continue
            except OSError:
                pass
        sentences.append(text)
    return sentences

def main(args):
    model = Translator(
        model_path=args.model,
        device=args.device,
        tokenizer_src=args.tokenizer_src,
        tokenizer_tgt=args.tokenizer_tgt,
        output_name=args.output_name
    )
    input_data = parse_input(args.input)
    if args.output:
        open(args.output, 'w').close()

    def sentences():
        if input_data:
            for sentence in input_data:
                sentence = sentence.strip()
                if sentence:
                    print("> {}".format(sentence))
                    yield sentence
        else:
            while True:
                yield input("> ")

    start_time = perf_counter()
    # loop on user's or prepared questions
    for sentence in sentences():
        if not sentence.strip():
            break

        try:
            translation = model(sentence)
            print(translation)
            if args.output:
                with open(args.output, 'a', encoding='utf8') as f:
                    print(translation, file=f)
        except Exception:
            log.error("an error occurred", exc_info=True)

    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)

#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation
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
import time
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore
from tokenizers import SentencePieceBPETokenizer


class Translator:
    """ Language translation.

    Arguments:
        model_xml (str): path to model's .xml file.
        model_bin (str): path to model's .bin file.
        tokenizer_src (str): path to src tokenizer.
        tokenizer_tgt (str): path to tgt tokenizer.
    """
    def __init__(self, model_xml, model_bin, device, tokenizer_src, tokenizer_tgt, output_name):
        self.model = TranslationEngine(model_xml, model_bin, device, output_name)
        self.max_tokens = self.model.get_max_tokens()
        self.tokenizer_src = Tokenizer(tokenizer_src, self.max_tokens)
        self.tokenizer_tgt = Tokenizer(tokenizer_tgt, self.max_tokens)

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
        translation = self.model(tokens)
        translation = self.tokenizer_tgt.decode(translation[0], remove_repeats)
        return translation


class TranslationEngine:
    """ OpenVINO engine for machine translation.

    Arguments:
        model_xml (str): path to model's .xml file.
        model_bin (str): path to model's .bin file.
        output_name (str): name of output blob of model.
    """
    def __init__(self, model_xml, model_bin, device, output_name):
        self.logger = log.getLogger("TranslationEngine")
        self.logger.info(f"model_xml: {model_xml}")
        self.logger.info(f"model_bin: {model_bin}")
        self.ie = IECore()
        self.net = self.ie.read_network(
            model=model_xml,
            weights=model_bin
        )
        self.logger.info("loading model to the {}".format(device))
        self.net_exec = self.ie.load_network(self.net, device)
        self.output_name = output_name
        assert self.output_name != "", "there is not output in model"

    def get_max_tokens(self):
        """ Get maximum number of tokens that supported by model.

        Returns:
            max_tokens (int): maximum number of tokens;
        """
        return self.net.input_info["tokens"].input_data.shape[1]

    def __call__(self, tokens):
        """ Inference method.

        Arguments:
            tokens (np.array): input sentence in tokenized format.

        Returns:
            translation (np.array): translated sentence in tokenized format.
        """
        out = self.net_exec.infer(
            inputs={"tokens": tokens}
        )
        return out[self.output_name]


class Tokenizer:
    """ Sentence tokenizer.

    Arguments:
        path (str): path to tokenizer's model folder.
        max_tokens (int): max tokens.
    """
    def __init__(self, path, max_tokens):
        self.logger = log.getLogger("Tokenizer")
        self.logger.info("loading tokenizer")
        self.logger.info(f"path: {path}")
        self.logger.info(f"max_tokens: {max_tokens}")
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
                        help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
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
    log.basicConfig(format="[ %(levelname)s ] [ %(name)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    logger = log.getLogger("main")
    logger.info("creating translator")
    model = Translator(
        model_xml=args.model,
        model_bin=args.model.with_suffix(".bin"),
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

    # loop on user's or prepared questions
    for sentence in sentences():
        if not sentence.strip():
            break

        try:
            start = time.perf_counter()
            translation = model(sentence)
            stop = time.perf_counter()
            print(translation)
            logger.info(f"time: {stop - start} s.")
            if args.output:
                with open(args.output, 'a', encoding='utf8') as f:
                    print(translation, file=f)
        except Exception:
            log.error("an error occurred", exc_info=True)


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)

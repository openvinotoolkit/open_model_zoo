#!/usr/bin/env python3

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

import logging as log
import re
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

from openvino.inference_engine import IECore, get_version

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs
from models import BertNamedEntityRecognition

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

sentence_splitter = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
label_to_tag = ['O', 'B-MIS', 'I-MIS', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-v", "--vocab", help="Required. path to the vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-i", "--input", help="Required. URL to a page with context",
                      action='append',
                      required=True, type=str)
    args.add_argument("--input_names",
                      help="Optional. Inputs names for the network. "
                           "Default values are \"input_ids,attention_mask,token_type_ids\" ",
                      required=False, type=str, default="input_ids,attention_mask,token_type_ids")
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU", default="CPU", type=str)
    return parser


def main():
    args = build_argparser().parse_args()

    paragraphs = get_paragraphs(args.input)

    preprocessing_start_time = perf_counter()
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))

    # Get context as a string (as we might need it's length for the sequence reshape)
    context = '\n'.join(paragraphs)
    sentences = re.split(sentence_splitter, context)
    preprocessed_sentences = [text_to_tokens(sentence, vocab) for sentence in sentences]
    max_sentence_length = max([len(tokens) + 2 for tokens, _ in preprocessed_sentences])
    preprocessing_total_time = (perf_counter() - preprocessing_start_time) * 1e3

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    log.info('Reading model {}'.format(args.model))
    model = BertNamedEntityRecognition(ie, args.model, vocab, args.input_names)
    if max_sentence_length > model.max_length:
        model.reshape(max_sentence_length)

    model_exec = ie.load_network(model.net, args.device)
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    start_time = perf_counter()
    for sentence, (c_tokens_id, c_token_s_e) in zip(sentences, preprocessed_sentences):
        inputs, meta = model.preprocess(c_tokens_id)
        raw_result = model_exec.infer(inputs)
        score, filtered_labels_id = model.postprocess(raw_result, meta)

        if not filtered_labels_id:
            continue

        log.info('\t\tSentence: \n\t{}'.format(sentence))
        visualized = set()
        for idx, label_idx in filtered_labels_id:
            word_s, word_e = c_token_s_e[idx - 1]
            if (word_s, word_e) in visualized:
                continue
            visualized.add((word_s, word_e))
            word = sentence[word_s:word_e]
            confidence = score[idx][label_idx]
            tag = label_to_tag[label_idx]
            log.info('\n\tWord: {}\n\tConfidence: {}\n\tTag: {}'.format(word, confidence, tag))

    total_latency = (perf_counter() - start_time) * 1e3 + preprocessing_total_time
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))


if __name__ == '__main__':
    sys.exit(main() or 0)

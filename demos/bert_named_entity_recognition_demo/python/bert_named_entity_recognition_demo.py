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
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs

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
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # load vocabulary file for model
    log.info("Loading vocab file:\t{}".format(args.vocab))
    vocab = load_vocab_file(args.vocab)
    log.info("{} tokens loaded".format(len(vocab)))

    # get context as a string (as we might need it's length for the sequence reshape)
    paragraphs = get_paragraphs(args.input)
    context = '\n'.join(paragraphs)
    log.info("Size: {} chars".format(len(context)))
    sentences = re.split(sentence_splitter, context)
    preprocessed_sentences = [text_to_tokens(sentence, vocab) for sentence in sentences]
    max_sent_length = max([len(tokens) + 2 for tokens, _ in preprocessed_sentences])

    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    # check input and output names
    input_names = [i.strip() for i in args.input_names.split(',')]
    if ie_encoder.input_info.keys() != set(input_names):
        log.error("Input names do not match")
        log.error("    The demo expects input names: {}. "
                  "Please use the --input_names to specify the right names "
                  "(see actual values below)".format(input_names))
        log.error("    Actual network input names: {}".format(list(ie_encoder.input_info.keys())))
        raise Exception("Unexpected network input names")
    if len(ie_encoder.outputs) != 1:
        log.log.error('Demo expects model with single output, while provided {}'.format(len(ie_encoder.outputs)))
        raise Exception('Unexpected number of outputs')
    output_names = list(ie_encoder.outputs)
    max_length = ie_encoder.input_info[input_names[0]].input_data.shape[1]
    if max_sent_length > max_length:
        input_shapes = {
            input_names[0]: [1, max_sent_length],
            input_names[1]: [1, max_sent_length],
            input_names[2]: [1, max_sent_length]
        }
        ie_encoder.reshape(input_shapes)
        max_length = max_sent_length
    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)
    # maximum number of tokens that can be processed by network at once
    t0 = time.perf_counter()
    t_count = 0

    def get_score(name):
        out = np.exp(res[name][0])
        return out / out.sum(axis=-1, keepdims=True)

    for sentence, (c_tokens_id, c_token_s_e) in zip(sentences, preprocessed_sentences):
        # form the request
        tok_cls = vocab['[CLS]']
        tok_sep = vocab['[SEP]']
        input_ids = [tok_cls] + c_tokens_id + [tok_sep]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # pad the rest of the request
        pad_len = max_length - len(input_ids)
        input_ids += [0] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        # create numpy inputs for IE
        inputs = {
            input_names[0]: np.array([input_ids], dtype=np.int32),
            input_names[1]: np.array([attention_mask], dtype=np.int32),
            input_names[2]: np.array([token_type_ids], dtype=np.int32),
        }
        if len(input_names)>3:
            inputs[input_names[3]] = np.arange(len(input_ids), dtype=np.int32)[None, :]

        t_start = time.perf_counter()
        # infer by IE
        res = ie_encoder_exec.infer(inputs=inputs)
        t_end = time.perf_counter()
        t_count += 1
        log.info("Sequence of length {} is processed with {:0.2f} requests/sec ({:0.2} sec per request)".format(
            max_length,
            1 / (t_end - t_start),
            t_end - t_start
        ))


        score = get_score(output_names[0])
        labels_idx = score.argmax(-1)
        filtered_labels_idx = [
            (idx, label_idx)
            for idx, label_idx in enumerate(labels_idx)
            if label_idx != 0 and 0 < idx < max_length - pad_len
        ]

        if not filtered_labels_idx:
            continue

        log.info('Sentence: \n\t{}'.format(sentence))
        visualized = set()
        for idx, label_idx in filtered_labels_idx:
            word_s, word_e = c_token_s_e[idx - 1]
            if (word_s, word_e) in visualized:
                continue
            visualized.add((word_s, word_e))
            word = sentence[word_s:word_e]
            log.info('\n\tWord: {}\n\tConfidence: {}\n\tTag: {}'.format(word, score[idx][label_idx], label_to_tag[label_idx]))

    t1 = time.perf_counter()
    log.info("The performance below is reported only for reference purposes, "
            "please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.")
    log.info("{} requests of {} length were processed in {:0.2f}sec ({:0.2}sec per request)".format(
        t_count,
        max_length,
        t1 - t0,
        (t1 - t0) / t_count
    ))


if __name__ == '__main__':
    sys.exit(main() or 0)

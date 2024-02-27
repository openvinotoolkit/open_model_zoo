#!/usr/bin/env python3

"""
 Copyright (c) 2021-2024 Intel Corporation

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
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
from openvino import Core, get_version, PartialShape, Dimension
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
from gpt2_text_prediction.generation_utils import (load_vocab_file, softmax, get_top_k_logits, get_top_p_logits,
                                                   process_logits, stop_criteria)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-v", "--vocab", help="Required. Path to the vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument("--merges", help="Required. Path to the merges file",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Optional. Input prompt", required=False, type=str, action='append')
    args.add_argument("--max_sample_token_num", help="Optional. Maximum number of tokens in generated sample",
                      default=40, required=False, type=int)
    args.add_argument("--top_k", help="Optional. Number of tokens with the highest probability "
                                      "which will be kept for generation",
                      default=0, required=False, type=int)
    args.add_argument("--top_p", help="Optional. Maximum probability, tokens with such a probability "
                                      "and lower will be kept for generation",
                      default=0.9, required=False, type=float)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on. "
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument('--dynamic_shape', action='store_true', help='Run model with dynamic input sequence. If not provided, input sequence will be padded to max_seq_len')
    args.add_argument('--max_seq_len', type=int, required=False, default=1024, help='Optional. Maximum sequence length for processing. Default value is 1024')
    return parser


def main():
    args = build_argparser().parse_args()

    # load vocabulary file for model
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))

    # create tokenizer
    tokenizer = Tokenizer(BPE.from_file(str(args.vocab), str(args.merges)))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # read IR
    log.info('Reading model {}'.format(args.model))
    model = core.read_model(args.model)

    # check number inputs and outputs
    if len(model.inputs) != 1:
        raise RuntimeError('The demo expects model with single input, while provided {}'.format(
            len(model.inputs)))
    if len(model.outputs) != 1:
        raise RuntimeError('The demo expects model with single output, while provided {}'.format(
            len(model.outputs)))
    input_tensor = model.inputs[0].any_name

    if not args.dynamic_shape and (model.inputs[0].partial_shape.is_dynamic or model.inputs[0].shape[1] != args.max_seq_len):
        model.reshape({input_tensor: PartialShape([Dimension(1), Dimension(args.max_seq_len)])})

    if args.dynamic_shape:
        model.reshape({input_tensor: PartialShape([Dimension(1), Dimension(0, args.max_seq_len)])})

    # load model to the device
    compiled_model = core.compile_model(model, args.device)
    output_tensor = compiled_model.outputs[0]
    infer_request = compiled_model.create_infer_request()
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    if args.input:
        def prompts():
            for prompt in args.input:
                log.info("Input prompt: {}".format(prompt))
                yield prompt
    else:
        def prompts():
            while True:
                yield input('Type input prompt (empty string to exit):')

    # loop on user's or prepared prompts
    for prompt in prompts():
        if not prompt.strip():
            break

        # encode input
        tokens = tokenizer.encode_batch([prompt])[0].ids
        input_ids = np.array([tokens], dtype=np.int32)

        # maximum number of tokens that can be processed by network at once
        max_length = args.max_seq_len

        eos_token_id = len(vocab) - 1

        cur_input_len = input_ids.shape[-1]

        # maximum number of tokens that will be generated
        max_sample_token_num = args.max_sample_token_num + cur_input_len

        t0 = time.perf_counter()
        t_count = 0

        while True:
            model_input = input_ids
            if not args.dynamic_shape:
                # pad the rest of the request
                pad_len = max_length - cur_input_len
                model_input = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)

            # create numpy inputs for OpenVINO runtime
            inputs = {
                input_tensor: model_input,
            }

            # infer by OpenVINO runtime
            t_start = time.perf_counter()
            outputs = infer_request.infer(inputs)[output_tensor]
            t_end = time.perf_counter()
            t_count += 1
            log.info("Sequence of length {} is processed with {:0.2f} requests/sec ({:0.2} sec per request)".format(
                model_input.shape[1], 1 / (t_end - t_start), t_end - t_start))

            next_token_logits = outputs[:, cur_input_len-1, :]

            # pre-process distribution
            next_token_scores = process_logits(input_ids, next_token_logits, eos_token_id)
            if args.top_k > 0:
                next_token_scores = get_top_k_logits(next_token_scores, args.top_k)

            if args.top_p < 1.0:
                next_token_scores = get_top_p_logits(next_token_scores, args.top_p)

            # get next token id
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)

            # update info for the next step
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)

            cur_input_len = input_ids.shape[-1]

            if stop_criteria(input_ids, min(max_length, max_sample_token_num), eos_token_id):
                break

        t1 = time.perf_counter()

        text = tokenizer.decode_batch(input_ids)[0]

        log.info("{} requests were processed in {:0.2f}sec ({:0.2}sec per request)".format(
            t_count, t1 - t0, (t1 - t0) / t_count))

        # print result
        log.info("GENERATED SEQUENCE: {}".format(text))


if __name__ == '__main__':
    sys.exit(main() or 0)

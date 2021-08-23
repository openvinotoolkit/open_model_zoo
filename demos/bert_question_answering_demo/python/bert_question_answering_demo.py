#!/usr/bin/env python3

"""
 Copyright (c) 2020-2021 Intel Corporation

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
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import numpy as np
from openvino.inference_engine import IECore, get_version

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from tokens_bert import text_to_tokens, load_vocab_file, ContextData
from html_reader import get_paragraphs
from models import BertQuestionAnswering

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


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
    args.add_argument("--questions", type=str, nargs='+', metavar='QUESTION', help="Optional. Prepared questions")
    args.add_argument("--input_names",
                      help="Optional. Inputs names for the network. "
                           "Default values are \"input_ids,attention_mask,token_type_ids\" ",
                      required=False, type=str, default="input_ids,attention_mask,token_type_ids")
    args.add_argument("--output_names",
                      help="Optional. Outputs names for the network. "
                           "Default values are \"output_s,output_e\" ",
                      required=False, type=str, default="output_s,output_e")
    args.add_argument("--model_squad_ver", help="Optional. SQUAD version used for model fine tuning",
                      default="1.2", required=False, type=str)
    args.add_argument("-q", "--max_question_token_num", help="Optional. Maximum number of tokens in question",
                      default=8, required=False, type=int)
    args.add_argument("-a", "--max_answer_token_num", help="Optional. Maximum number of tokens in answer",
                      default=15, required=False, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument('-r', '--reshape', action='store_true',
                      help="Optional. Auto reshape sequence length to the "
                           "input context + max question len (to improve the speed)")
    args.add_argument('-c', '--colors', action='store_true',
                      help="Optional. Nice coloring of the questions/answers. "
                           "Might not work on some terminals (like Windows* cmd console)")
    return parser


# return entire sentence as start-end positions for a given answer (within the sentence)
def find_sentence_range(context, s, e):
    # find start of sentence
    for c_s in range(s, max(-1, s - 200), -1):
        if context[c_s] in "\n.":
            c_s += 1
            break
    # find end of sentence
    for c_e in range(max(0, e - 1), min(len(context), e + 200), +1):
        if context[c_e] in "\n.":
            break
    return c_s, c_e


def main():
    args = build_argparser().parse_args()

    paragraphs = get_paragraphs(args.input)

    preprocessing_start_time = perf_counter()
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))

    # get context as a string (as we might need it's length for the sequence reshape)
    context = '\n'.join(paragraphs)
    # encode context into token ids list
    c_tokens_id, c_tokens_se = text_to_tokens(context.lower(), vocab)
    total_latency = (perf_counter() - preprocessing_start_time) * 1e3

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    log.info('Reading model {}'.format(args.model))
    model = BertQuestionAnswering(ie, args.model, vocab, args.input_names, args.output_names,
                                  args.max_answer_token_num, args.model_squad_ver)
    if args.reshape:
        # find the closest multiple of 64, if it is smaller than current network's sequence length, do reshape
        new_length = min(model.max_length, int(np.ceil((len(c_tokens_id) + args.max_question_token_num) / 64) * 64))
        if new_length < model.max_length:
            model.reshape(new_length)
        else:
            log.debug("\tSkipping network reshaping,"
                      " as (context length + max question length) exceeds the current (input) network sequence length")

    model_exec = ie.load_network(model.net, args.device)
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    if args.questions:
        def questions():
            for question in args.questions:
                log.info("\n\tQuestion: {}".format(question))
                yield question
    else:
        def questions():
            while True:
                yield input('\n\tType a question (empty string to exit): ')

    for question in questions():
        if not question.strip():
            break

        start_time = perf_counter()
        q_tokens_id, _ = text_to_tokens(question.lower(), vocab)
        answers = []

        # calculate number of tokens for context in each inference request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        context_window_len = model.max_length - (len(q_tokens_id) + 3)

        # token num between two neighbour context windows
        # 1/2 means that context windows are overlapped by half
        c_stride = context_window_len // 2

        # init a window to iterate over context
        c_s, c_e = 0, min(context_window_len, len(c_tokens_id))

        # iterate while context window is not empty
        while c_e > c_s:
            c_data = ContextData(c_tokens_id[c_s:c_e], c_tokens_se[c_s:c_e], context=context)
            inputs, meta = model.preprocess((c_data, q_tokens_id))
            raw_result = model_exec.infer(inputs)
            output = model.postprocess(raw_result, meta)

            # update answers list
            same = [i for i, ans in enumerate(answers) if ans[1:3] == output[1:3]]
            if not same:
                answers.append(output)
            else:
                assert len(same) == 1
                prev_score = answers[same[0]][0]
                answers[same[0]] = (max(output[0], prev_score), *output[1:])

            if c_e == len(c_tokens_id):
                break

            c_s = min(c_s + c_stride, len(c_tokens_id))
            c_e = min(c_s + context_window_len, len(c_tokens_id))

        def mark(txt):
            return "\033[91m" + txt + "\033[0m" if args.colors else "*" + txt + "*"

        log.info("\t\tShow top 3 answers")
        answers = sorted(answers, key=lambda x: -x[0])[:3]
        for score, s, e in answers:
            c_s, c_e = find_sentence_range(context, s, e)
            context_str = context[c_s:s] + mark(context[s:e]) + context[e:c_e]
            log.info("Answer: {}\n\t Score: {:0.2f}\n\t Context: {}".format(mark(context[s:e]), score, context_str))

        total_latency += (perf_counter() - start_time) * 1e3

    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))


if __name__ == '__main__':
    sys.exit(main() or 0)

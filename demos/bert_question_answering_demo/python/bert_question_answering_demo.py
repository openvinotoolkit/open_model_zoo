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

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

from html_reader import get_paragraphs

from model_api.models import BertQuestionAnswering
from model_api.models.tokens_bert import text_to_tokens, load_vocab_file, ContextWindow
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-v", "--vocab", help="Required. Path to the vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument('-m', '--model', required=True,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using OVMS adapter.')
    args.add_argument("-i", "--input", help="Required. URL to a page with context",
                      action='append',
                      required=True, type=str)
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument("--questions", type=str, nargs='+', metavar='QUESTION', help="Optional. Prepared questions")
    args.add_argument("--input_names",
                      help="Optional. Inputs names for the network. "
                           "Default values are \"input_ids,attention_mask,token_type_ids\" ",
                      required=False, type=str, default="input_ids,attention_mask,token_type_ids")
    args.add_argument('--layout', type=str, default=None,
                      help='Optional. Model inputs layouts. '
                           'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
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
    args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests.',
                      default=0, type=int)
    args.add_argument('-nstreams', '--num_streams',
                      help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                           'mode (for HETERO and MULTI device cases use format '
                           '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                      default='', type=str)
    args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                      help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')
    return parser


def update_answers_list(answers, output):
    same = [i for i, ans in enumerate(answers) if ans[1:3] == output[1:3]]
    if not same:
        answers.append(output)
    else:
        assert len(same) == 1
        prev_score = answers[same[0]][0]
        answers[same[0]] = (max(output[0], prev_score), *output[1:])


class Visualizer:
    def __init__(self, context, use_colors):
        self.context = context
        self.use_colors = use_colors

    def mark(self, text):
        return "\033[91m" + text + "\033[0m" if self.use_colors else "*" + text + "*"

    # return entire sentence as start-end positions for a given answer (within the sentence)
    def find_sentence_range(self, s, e):
        # find start of sentence
        for c_s in range(s, max(-1, s - 200), -1):
            if self.context[c_s] in "\n.":
                c_s += 1
                break
        # find end of sentence
        for c_e in range(max(0, e - 1), min(len(self.context), e + 200), +1):
            if self.context[c_e] in "\n.":
                break
        return c_s, c_e

    def show_answers(self, answers):
        log.info("\t\tShow top 3 answers")
        answers = sorted(answers, key=lambda x: -x[0])[:3]
        for score, s, e in answers:
            c_s, c_e = self.find_sentence_range(s, e)
            marked_answer = self.mark(self.context[s:e])
            context_str = self.context[c_s:s] + marked_answer + self.context[e:c_e]
            log.info("Answer: {}\n\t Score: {:0.2f}\n\t Context: {}".format(marked_answer, score, context_str))


class ContextSource:
    def __init__(self, q_tokens_id, c_tokens, model_max_length):
        self.q_tokens_id = q_tokens_id

        # calculate number of tokens for context in each inference request.
        # reserve 3 positions for special tokens: [CLS] q_tokens [SEP] c_tokens [SEP]
        c_window_len = model_max_length - (len(self.q_tokens_id) + 3)

        self.window = ContextWindow(c_window_len, *c_tokens)

    def get_data(self):
        c_data = self.window.get_context_data()
        self.window.move()
        return (c_data, self.q_tokens_id)

    def is_over(self):
        return self.window.is_over()


def main():
    args = build_argparser().parse_args()

    paragraphs = get_paragraphs(args.input)

    preprocessing_start_time = perf_counter()
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))

    # get context as a string (as we might need it's length for the sequence reshape)
    context = '\n'.join(paragraphs)
    visualizer = Visualizer(context, args.colors)
    # encode context into token ids list
    c_tokens = text_to_tokens(context.lower(), vocab)
    total_latency = (perf_counter() - preprocessing_start_time) * 1e3

    if args.adapter == 'openvino':
        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
    elif args.adapter == 'ovms':
        model_adapter = OVMSAdapter(args.model)

    config = {
        'vocab': vocab,
        'input_names': args.input_names,
        'output_names': args.output_names,
        'max_answer_token_num': args.max_answer_token_num,
        'squad_ver': args.model_squad_ver
    }
    model = BertQuestionAnswering(model_adapter, config)
    if args.reshape:
        # find the closest multiple of 64, if it is smaller than current network's sequence length, do reshape
        new_length = min(model.max_length, int(np.ceil((len(c_tokens[0]) + args.max_question_token_num) / 64) * 64))
        if new_length < model.max_length:
            try:
                model.reshape(new_length)
            except RuntimeError:
                log.error("Failed to reshape the network, please retry the demo without '-r' option")
                sys.exit(-1)
        else:
            log.debug("\tSkipping network reshaping,"
                      " as (context length + max question length) exceeds the current (input) network sequence length")
    model.log_layers_info()

    pipeline = AsyncPipeline(model)

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

        answers = []
        next_window_id = 0
        next_window_id_to_show = 0
        start_time = perf_counter()
        q_tokens_id, _ = text_to_tokens(question.lower(), vocab)
        source = ContextSource(q_tokens_id, c_tokens, model.max_length)

        while True:
            if pipeline.callback_exceptions:
                raise pipeline.callback_exceptions[0]
            results = pipeline.get_result(next_window_id_to_show)
            if results:
                next_window_id_to_show += 1
                update_answers_list(answers, results[0])
                continue

            if pipeline.is_ready():
                if source.is_over():
                    break
                pipeline.submit_data(source.get_data(), next_window_id)
                next_window_id += 1
            else:
                pipeline.await_any()

        pipeline.await_all()
        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        for window_id in range(next_window_id_to_show, next_window_id):
            results = pipeline.get_result(window_id)
            update_answers_list(answers, results[0])

        visualizer.show_answers(answers)
        total_latency += (perf_counter() - start_time) * 1e3

    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))


if __name__ == '__main__':
    sys.exit(main() or 0)

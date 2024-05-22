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

from model_api.models import BertEmbedding, BertQuestionAnswering
from model_api.models.tokens_bert import text_to_tokens, load_vocab_file, ContextWindow
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Urls to a wiki pages with context",
                      action='append',
                      required=True, type=str)
    args.add_argument("--questions", type=str, nargs='+', metavar='QUESTION', help="Optional. Prepared questions")
    args.add_argument("--best_n",
                      help="Optional. Number of best (closest) contexts selected",
                      default=10,
                      required=False, type=int)
    args.add_argument("-v", "--vocab",
                      help="Required. Path to vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument("-m_emb", "--model_emb",
                      help="Required. Path to an .xml file with a trained model to build embeddings",
                      required=True, type=Path)
    args.add_argument("--input_names_emb",
                      help="Optional. Names for inputs in MODEL_EMB network. "
                           "For example 'input_ids,attention_mask,token_type_ids','position_ids'",
                      default='input_ids,attention_mask,token_type_ids,position_ids',
                      required=False, type=str)
    args.add_argument('--layout_emb', type=str, default=None,
                      help='Optional. MODEL_EMB inputs layouts. '
                           'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    args.add_argument("-m_qa", "--model_qa",
                      help="Optional. Path to an .xml file with a trained model to give exact answer",
                      default = None,
                      required=False, type=Path)
    args.add_argument("--input_names_qa",
                      help="Optional. Names for inputs in MODEL_QA network. "
                           "For example 'input_ids,attention_mask,token_type_ids','position_ids'",
                      default='input_ids,attention_mask,token_type_ids,position_ids',
                      required=False, type=str)
    args.add_argument("--output_names_qa",
                      help="Optional. Names for outputs in MODEL_QA network. For example 'output_s,output_e'",
                      default='output_s,output_e',
                      required=False, type=str)
    args.add_argument("--model_qa_squad_ver", help="Optional. SQUAD version used for QuestionAnswering model fine tuning",
                      default="1.2", required=False, type=str)
    args.add_argument('--layout_qa', type=str, default=None,
                      help='Optional. MODEL_QA inputs layouts. '
                           'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    args.add_argument("-a", "--max_answer_token_num",
                      help="Optional. Maximum number of tokens in exact answer",
                      default=15,
                      required=False, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU is acceptable. "
                           "The demo will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU",
                      required=False, type=str)
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


def update_answers_list(answers, output, c_data):
    same = [i for i, ans in enumerate(answers) if ans[1:3] == output[1:3] and ans[3] in c_data.context]
    if not same:
        answers.append((*output, c_data.context))
    else:
        assert len(same) == 1
        prev_score = answers[same[0]][0]
        answers[same[0]] = (max(output[0], prev_score), *output[1:], c_data.context)


class Visualizer:
    def __init__(self, use_colors):
        self.use_colors = use_colors

    def mark(self, text):
        return "\033[91m" + text + "\033[0m" if self.use_colors else "*" + text + "*"

    @staticmethod
    def show_closest_contexts(distances_filtered):
        for (dist, c_data) in distances_filtered:
            log.info("\n\t Embedding distance: {}\n\t Context: {}".format(dist, c_data.context.strip()))
        print('\n')

    def show_answers(self, answers):
        for score, s, e, context in answers:
            marked_answer = self.mark(context[s:e])
            context_str = context[:s] + marked_answer + context[e:]
            log.info("Answer: {}\n\t Score: {:0.2f}\n\t Context: {}".format(marked_answer, score, context_str))


class ContextSource:
    def __init__(self, paragraphs, vocab, c_window_len):
        self.paragraphs = paragraphs
        self.c_tokens = [text_to_tokens(par.lower(), vocab) for par in paragraphs]
        self.c_window_len = c_window_len
        self.par_id = -1
        self.get_next_paragraph()

    def get_data(self):
        c_data = self.window.get_context_data(self.paragraphs[self.par_id])
        self.window.move()
        if self.window.is_over():
            self.get_next_paragraph()
        return c_data

    def get_next_paragraph(self):
        self.par_id += 1
        if not self.is_over():
            if self.c_tokens[self.par_id][0]:
                self.window = ContextWindow(self.c_window_len, *self.c_tokens[self.par_id])
            else:
                self.get_next_paragraph()

    def is_over(self):
        return self.par_id == len(self.paragraphs)


def main():
    args = build_argparser().parse_args()

    paragraphs = get_paragraphs(args.input)

    vocab_start_time = perf_counter()
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))
    visualizer = Visualizer(args.colors)
    total_latency = (perf_counter() - vocab_start_time) * 1e3

    core = create_core()
    plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
    model_emb_adapter = OpenvinoAdapter(core, args.model_emb, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout_emb})
    model_emb = BertEmbedding(model_emb_adapter, {'vocab': vocab, 'input_names': args.input_names_emb})
    model_emb.log_layers_info()

    # reshape BertEmbedding model to infer short questions and long contexts
    max_len_context = 384
    max_len_question = 32

    for new_length in [max_len_question, max_len_context]:
        model_emb.reshape(new_length)
        if new_length == max_len_question:
            emb_request = core.compile_model(model_emb_adapter.model, args.device).create_infer_request()
        else:
            emb_pipeline = AsyncPipeline(model_emb)

    if args.model_qa:
        model_qa_adapter = OpenvinoAdapter(core, args.model_qa, device=args.device, plugin_config=plugin_config,
                                           max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout_qa})
        config = {
            'vocab': vocab,
            'input_names': args.input_names_qa,
            'output_names': args.output_names_qa,
            'max_answer_token_num': args.max_answer_token_num,
            'squad_ver': args.model_qa_squad_ver
        }
        model_qa = BertQuestionAnswering(model_qa_adapter, config)
        model_qa.log_layers_info()
        qa_pipeline = AsyncPipeline(model_qa)

    log.info("\t\tStage 1    (Calc embeddings for the context)")
    contexts_all = []
    start_time = perf_counter()

    # get context as string and then encode it into token id list
    # calculate number of tokens for context in each request.
    # reserve 3 positions for special tokens [CLS] q_tokens [SEP] c_tokens [SEP]
    if args.model_qa:
        # to make context be able to pass model_qa together with question
        c_window_len = model_qa.max_length - (max_len_question + 3)
    else:
        # to make context be able to pass model_emb without question
        c_window_len = max_len_context - 2

    def calc_question_embedding(tokens_id):
        num = min(max_len_question - 2, len(tokens_id))
        inputs, _ = model_emb.preprocess((tokens_id[:num], max_len_question))
        emb_request.infer(inputs)
        raw_result = model_emb_adapter.get_raw_result(emb_request)
        return model_emb.postprocess(raw_result, None)

    source = ContextSource(paragraphs, vocab, c_window_len)
    next_window_id = 0
    next_window_id_to_show = 0
    contexts_all = []

    while True:
        if emb_pipeline.callback_exceptions:
            raise emb_pipeline.callback_exceptions[0]
        results = emb_pipeline.get_result(next_window_id_to_show)
        if results:
            embedding, meta = results
            meta['c_data'].emb = embedding
            contexts_all.append(meta['c_data'])
            next_window_id_to_show += 1
            continue

        if emb_pipeline.is_ready():
            if source.is_over():
                break
            c_data = source.get_data()
            num = min(max_len_context - 2, len(c_data.c_tokens_id))
            emb_pipeline.submit_data((c_data.c_tokens_id[:num], max_len_context), next_window_id, {'c_data': c_data})
            next_window_id += 1
        else:
            emb_pipeline.await_any()

    emb_pipeline.await_all()
    if emb_pipeline.callback_exceptions:
        raise emb_pipeline.callback_exceptions[0]
    for window_id in range(next_window_id_to_show, next_window_id):
        results = emb_pipeline.get_result(window_id)
        embedding, meta = results
        meta['c_data'].emb = embedding
        contexts_all.append(meta['c_data'])
        next_window_id_to_show += 1

    total_latency += (perf_counter() - start_time) * 1e3
    context_embeddings_time = total_latency

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
        log.info("\t\tStage 2    (Calc question embedding and compare with {} context embeddings)".format(len(contexts_all)))
        q_tokens_id, _ = text_to_tokens(question.lower(), vocab)
        q_emb = calc_question_embedding(q_tokens_id)
        distances = [(np.linalg.norm(context.emb - q_emb, 2), context) for context in contexts_all]
        distances.sort(key=lambda x: x[0])
        keep_num = min(args.best_n, len(distances))
        distances_filtered = distances[:keep_num]

        log.info("The closest {} contexts to question filtered from {} context embeddings:".format(keep_num, len(distances)))
        visualizer.show_closest_contexts(distances_filtered)

        if args.model_qa:
            answers = []
            next_context_id = 0
            next_context_id_to_show = 0

            while True:
                if qa_pipeline.callback_exceptions:
                    raise qa_pipeline.callback_exceptions[0]
                results = qa_pipeline.get_result(next_context_id_to_show)
                if results:
                    next_context_id_to_show += 1
                    output, meta = results
                    update_answers_list(answers, output, meta['c_data'])
                    continue

                if qa_pipeline.is_ready():
                    if next_context_id == len(distances_filtered):
                        break
                    _, c_data = distances_filtered[next_context_id]
                    qa_pipeline.submit_data((c_data, q_tokens_id), next_context_id, {'c_data': c_data})
                    next_context_id += 1
                else:
                    qa_pipeline.await_any()

            qa_pipeline.await_all()
            if qa_pipeline.callback_exceptions:
                raise qa_pipeline.callback_exceptions[0]
            for context_id in range(next_context_id_to_show, next_context_id):
                results = qa_pipeline.get_result(context_id)
                output, meta = results
                update_answers_list(answers, output, meta['c_data'])

            log.info("\t\tStage 3    (Show top 3 answers from {} closest contexts of Stage 1)".format(len(answers)))
            answers = sorted(answers, key=lambda x: -x[0])[:3]
            visualizer.show_answers(answers)

        total_latency += (perf_counter() - start_time) * 1e3

    log.info("Metrics report:")
    log.info("\tContext embeddings latency (stage 1): {:.1f} ms".format(context_embeddings_time))
    log.info("\tLatency (all stages): {:.1f} ms".format(total_latency))


if __name__ == '__main__':
    sys.exit(main() or 0)

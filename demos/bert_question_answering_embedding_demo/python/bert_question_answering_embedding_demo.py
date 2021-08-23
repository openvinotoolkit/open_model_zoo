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
from models import BertEmbedding, BertQuestionAnswering

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
    return parser


def main():
    args = build_argparser().parse_args()

    paragraphs = get_paragraphs(args.input)

    vocab_start_time = perf_counter()
    vocab = load_vocab_file(args.vocab)
    log.debug("Loaded vocab file from {}, get {} tokens".format(args.vocab, len(vocab)))
    total_latency = (perf_counter() - vocab_start_time) * 1e3

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    log.info('Reading Bert Embedding model {}'.format(args.model_emb))
    model_emb = BertEmbedding(ie, args.model_emb, vocab, args.input_names_emb)

    # reshape Bert Embedding model to infer short questions and long contexts
    model_emb_exec_dict = {}
    max_len_context = 384
    max_len_question = 32

    for new_length in [max_len_question, max_len_context]:
        model_emb.reshape(new_length)
        model_emb_exec_dict[new_length] = ie.load_network(model_emb.net, args.device)
    log.info('The Bert Embedding model {} is loaded to {}'.format(args.model_emb, args.device))

    if args.model_qa:
        log.info('Reading Question Answering model {}'.format(args.model_qa))
        model_qa = BertQuestionAnswering(ie, args.model_qa, vocab, args.input_names_qa, args.output_names_qa,
                                         args.max_answer_token_num, args.model_qa_squad_ver)
        model_qa_exec = ie.load_network(model_qa.net, args.device)
        log.info('The Question Answering model {} is loaded to {}'.format(args.model_qa, args.device))

    def calc_embedding(model, tokens_id, max_length):
        model_emb_exec = model_emb_exec_dict[max_length]
        num = min(max_length - 2, len(tokens_id))
        inputs, _ = model.preprocess((tokens_id[:num], max_length))
        raw_result = model_emb_exec.infer(inputs)
        return model.postprocess(raw_result, {})

    log.info("\t\tStage 1    (Calc embeddings for the context)")
    contexts_all = []
    start_time = perf_counter()
    for paragraph in paragraphs:
        c_tokens_id, c_tokens_se = text_to_tokens(paragraph.lower(), vocab)
        if not c_tokens_id:
            continue

        # get context as string and then encode it into token id list
        # calculate number of tokens for context in each request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        if args.model_qa:
            #to make context be able to pass model_qa together with question
            c_wnd_len = model_qa.max_length - (max_len_question + 3)
        else:
            #to make context be able to pass model_emb without question
            c_wnd_len = max_len_context - 2

        # token num between 2 neighbours context windows
        # 1/2 means that context windows are interleaved by half
        c_stride = c_wnd_len // 2

        # init scan window
        c_s, c_e = 0, min(c_wnd_len, len(c_tokens_id))

        # iterate while context window is not empty
        while c_e > c_s:
            c_data = ContextData(c_tokens_id[c_s:c_e], c_tokens_se[c_s:c_e], context=paragraph,
                                 c_emb=calc_embedding(model_emb, c_tokens_id[c_s:c_e], max_len_context))
            contexts_all.append(c_data)

            if c_e == len(c_tokens_id):
                break

            c_s, c_e = c_s + c_stride, c_e + c_stride

            shift_left = max(0, c_e - len(c_tokens_id))
            c_s, c_e = c_s - shift_left, c_e - shift_left
            assert c_s >= 0, "start can be left of 0 only with window less than len but in this case we can not be here"
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

        q_emb = calc_embedding(model_emb, q_tokens_id, max_len_question)
        distances = [(np.linalg.norm(context.c_emb - q_emb, 2), context) for context in contexts_all]
        distances.sort(key=lambda x: x[0])
        keep_num = min(args.best_n, len(distances))
        distances_filtered = distances[:keep_num]

        log.info("The closest {} contexts to question filtered from {} context embeddings:".format(keep_num, len(distances)))
        for (dist, c_data) in distances_filtered:
            log.info("\n\t Embedding distance: {}\n\t Context: {}".format(dist, c_data.context.strip()))
        print('\n')

        if args.model_qa:

            answers = []

            for _, c_data in distances_filtered:
                inputs, meta = model_qa.preprocess((c_data, q_tokens_id))
                raw_result = model_qa_exec.infer(inputs)
                output = model_qa.postprocess(raw_result, meta)

                # update answers list
                same = [i for i, ans in enumerate(answers) if ans[1:3] == output[1:3] and ans[3] in c_data.context]
                if not same:
                    answers.append((*output, c_data.context))
                else:
                    assert len(same) == 1
                    prev_score = answers[same[0]][0]
                    answers[same[0]] = (max(output[0], prev_score), *output[1:], c_data.context)

            def mark(txt):
                return "\033[91m" + txt + "\033[0m" if args.colors else "*" + txt + "*"

            answers = sorted(answers, key=lambda x: -x[0])[:3]
            log.info("\t\tStage 3    (Show top 3 answers from {} closest contexts of Stage 1)".format(len(answers)))
            for score, s, e, context in answers:
                context_str = context[:s] + mark(context[s:e]) + context[e:]
                log.info("Answer: {}\n\t Score: {:0.2f}\n\t Context: {}".format(mark(context[s:e]), score, context_str))

        total_latency += (perf_counter() - start_time) * 1e3

    log.info("Metrics report:")
    log.info("\tLatency (all stages): {:.1f} ms".format(total_latency))
    log.info("\tContext embeddings latency (stage 1): {:.1f} ms".format(context_embeddings_time))


if __name__ == '__main__':
    sys.exit(main() or 0)

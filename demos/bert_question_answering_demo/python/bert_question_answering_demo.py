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
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs


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

# return entire sentence as start-end positions for a given answer (within the sentence).
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
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    if args.colors:
        COLOR_RED = "\033[91m"
        COLOR_RESET = "\033[0m"
    else:
        COLOR_RED = ""
        COLOR_RESET = ""

    # load vocabulary file for model
    log.info("Loading vocab file:\t{}".format(args.vocab))
    vocab = load_vocab_file(args.vocab)
    log.info("{} tokens loaded".format(len(vocab)))

    # get context as a string (as we might need it's length for the sequence reshape)
    paragraphs = get_paragraphs(args.input)
    context = '\n'.join(paragraphs)
    log.info("Size: {} chars".format(len(context)))
    log.info("Context: " + COLOR_RED + context + COLOR_RESET)
    # encode context into token ids list
    c_tokens_id, c_tokens_se = text_to_tokens(context.lower(), vocab)

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

    if args.reshape:
        # reshape the sequence length to the context + maximum question length (in tokens)
        first_input_layer = next(iter(ie_encoder.input_info))
        c = ie_encoder.input_info[first_input_layer].input_data.shape[1]
        # find the closest multiple of 64, if it is smaller than current network's sequence length, let' use that
        seq = min(c, int(np.ceil((len(c_tokens_id) + args.max_question_token_num) / 64) * 64))
        if seq < c:
            new_shapes = {}
            for input_name, input_info in ie_encoder.input_info.items():
                n, c = input_info.input_data.shape
                new_shapes[input_name] = [n, seq]
                log.info("Reshaped input {} from {} to the {}".format(
                    input_name, input_info.input_data.shape, new_shapes[input_name]))
            log.info("Attempting to reshape the network to the modified inputs...")
            try:
                ie_encoder.reshape(new_shapes)
                log.info("Successful!")
            except RuntimeError:
                log.error("Failed to reshape the network, please retry the demo without '-r' option")
                sys.exit(-1)
        else:
            log.info("Skipping network reshaping,"
                     " as (context length + max question length) exceeds the current (input) network sequence length")

    # check input and output names
    input_names = [i.strip() for i in args.input_names.split(',')]
    output_names = [o.strip() for o in args.output_names.split(',')]
    if ie_encoder.input_info.keys() != set(input_names) or ie_encoder.outputs.keys() != set(output_names):
        log.error("Input or Output names do not match")
        log.error("    The demo expects input->output names: {}->{}. "
                  "Please use the --input_names and --output_names to specify the right names "
                  "(see actual values below)".format(input_names, output_names))
        log.error("    Actual network input->output names: {}->{}".format(list(ie_encoder.input_info.keys()),
                                                                          list(ie_encoder.outputs.keys())))
        raise Exception("Unexpected network input or output names")

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    if args.questions:
        def questions():
            for question in args.questions:
                log.info("Question: {}".format(question))
                yield question
    else:
        def questions():
            while True:
                yield input('Type question (empty string to exit):')

    # loop on user's or prepared questions
    for question in questions():
        if not question.strip():
            break

        q_tokens_id, _ = text_to_tokens(question.lower(), vocab)

        # maximum number of tokens that can be processed by network at once
        max_length = ie_encoder.input_info[input_names[0]].input_data.shape[1]

        # calculate number of tokens for context in each inference request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        c_wnd_len = max_length - (len(q_tokens_id) + 3)

        # token num between two neighbour context windows
        # 1/2 means that context windows are overlapped by half
        c_stride = c_wnd_len // 2

        t0 = time.perf_counter()
        t_count = 0

        # array of answers from each window
        answers = []

        # init a window to iterate over context
        c_s, c_e = 0, min(c_wnd_len, len(c_tokens_id))

        # iterate while context window is not empty
        while c_e > c_s:
            # form the request
            tok_cls = vocab['[CLS]']
            tok_sep = vocab['[SEP]']
            input_ids = [tok_cls] + q_tokens_id + [tok_sep] + c_tokens_id[c_s:c_e] + [tok_sep]
            token_type_ids = [0] + [0] * len(q_tokens_id) + [0] + [1] * (c_e - c_s) + [0]
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

            # get start-end scores for context
            def get_score(name):
                out = np.exp(res[name].reshape((max_length,)))
                return out / out.sum(axis=-1)

            score_s = get_score(output_names[0])
            score_e = get_score(output_names[1])

            # get 'no-answer' score (not valid if model has been fine-tuned on squad1.x)
            if args.model_squad_ver.split('.')[0] == '1':
                score_na = 0
            else:
                score_na = score_s[0] * score_e[0]

            # find product of all start-end combinations to find the best one
            c_s_idx = len(q_tokens_id) + 2  # index of first context token in tensor
            c_e_idx = max_length - (1 + pad_len)  # index of last+1 context token in tensor
            score_mat = np.matmul(
                score_s[c_s_idx:c_e_idx].reshape((c_e - c_s, 1)),
                score_e[c_s_idx:c_e_idx].reshape((1, c_e - c_s))
            )
            # reset candidates with end before start
            score_mat = np.triu(score_mat)
            # reset long candidates (>max_answer_token_num)
            score_mat = np.tril(score_mat, args.max_answer_token_num - 1)
            # find the best start-end pair
            max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
            max_score = score_mat[max_s, max_e] * (1 - score_na)

            # convert to context text start-end index
            max_s = c_tokens_se[c_s + max_s][0]
            max_e = c_tokens_se[c_s + max_e][1]

            # check that answers list does not have duplicates (because of context windows overlapping)
            same = [i for i, a in enumerate(answers) if a[1] == max_s and a[2] == max_e]
            if same:
                assert len(same) == 1
                # update existing answer record
                a = answers[same[0]]
                answers[same[0]] = (max(max_score, a[0]), max_s, max_e)
            else:
                # add new record
                answers.append((max_score, max_s, max_e))

            # check that context window reached the end
            if c_e == len(c_tokens_id):
                break

            # move to next window position
            c_s = min(c_s + c_stride, len(c_tokens_id))
            c_e = min(c_s + c_wnd_len, len(c_tokens_id))

        t1 = time.perf_counter()
        log.info("The performance below is reported only for reference purposes, "
                 "please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.")
        log.info("{} requests of {} length were processed in {:0.2f}sec ({:0.2}sec per request)".format(
            t_count,
            max_length,
            t1 - t0,
            (t1 - t0) / t_count
        ))

        # print top 3 results
        answers = sorted(answers, key=lambda x: -x[0])
        for score, s, e in answers[:3]:
            log.info("---answer: {:0.2f} {}".format(score, context[s:e]))
            c_s, c_e = find_sentence_range(context, s, e)
            log.info("   " + context[c_s:s] + COLOR_RED + context[s:e] + COLOR_RESET + context[e:c_e])


if __name__ == '__main__':
    sys.exit(main() or 0)

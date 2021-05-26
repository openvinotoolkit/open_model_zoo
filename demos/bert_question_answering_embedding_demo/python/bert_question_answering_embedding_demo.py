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
import sys
import time
import logging as log
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
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating Inference Engine")
    ie = IECore()

    #read model to calculate embedding
    model_xml_emb = args.model_emb
    model_bin_emb = model_xml_emb.with_suffix(".bin")

    log.info("Loading embedding network files:\n\t{}\n\t{}".format(model_xml_emb, model_bin_emb))
    ie_encoder_emb = ie.read_network(model=model_xml_emb, weights=model_bin_emb)
    input_names_model_emb = list(ie_encoder_emb.input_info.keys())
    input_names_emb = args.input_names_emb.split(',')
    log.info("Expected embedding input names: {}".format(input_names_emb))
    log.info("Network embedding input names: {}".format(input_names_model_emb))
    # check input names
    if set(input_names_model_emb) != set(input_names_emb):
        log.error("Unexpected embedding network input names")
        raise Exception("Unexpected embedding network input names")

    # check outputs
    output_names_model_emb = list(ie_encoder_emb.outputs.keys())
    if len(output_names_model_emb)>1:
        log.error("Expected only single output in embedding network but {} outputs detected".format(output_names_model_emb))
        raise Exception("Unexpected number of embedding network outputs")


    #reshape embedding model to infer short questions and long contexts
    ie_encoder_exec_emb_dict = {}
    max_length_c = 384
    max_length_q = 32

    for length in [max_length_q, max_length_c]:
        new_shapes = {}
        for i, input_info in ie_encoder_emb.input_info.items():
            new_shapes[i] = [1, length]
            log.info("Reshaped input {} from {} to the {}".format(
                i,
                input_info.input_data.shape,
                new_shapes[i]))
        log.info("Attempting to reshape the context embedding network to the modified inputs...")

        try:
            ie_encoder_emb.reshape(new_shapes)
            log.info("Successful!")
        except RuntimeError:
            log.error("Failed to reshape the embedding network")
            raise

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        ie_encoder_exec_emb_dict[length] = ie.load_network(network=ie_encoder_emb, device_name=args.device)

    # Read model for final exact qa
    if args.model_qa:
        model_xml = args.model_qa
        model_bin = model_xml.with_suffix(".bin")
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))

        ie_encoder_qa = ie.read_network(model=model_xml, weights=model_bin)
        ie_encoder_qa.batch_size = 1

        input_names_qa = args.input_names_qa.split(',')
        output_names_qa = args.output_names_qa.split(',')
        log.info("Expected input->output names: {}->{}".format(input_names_qa, output_names_qa))

        #check input and output names
        input_names_model_qa = list(ie_encoder_qa.input_info.keys())
        output_names_model_qa = list(ie_encoder_qa.outputs.keys())
        log.info("Network input->output names: {}->{}".format(input_names_model_qa, output_names_model_qa))
        if set(input_names_model_qa) != set(input_names_qa) or set(output_names_model_qa) != set(output_names_qa):
            log.error("Unexpected network input or output names")
            raise Exception("Unexpected network input or output names")

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        ie_encoder_qa_exec = ie.load_network(network=ie_encoder_qa, device_name=args.device)

        max_length_qc = ie_encoder_qa.input_info[input_names_qa[0]].input_data.shape[1]

    #load vocabulary file for all models
    log.info("Loading vocab file:\t{}".format(args.vocab))
    vocab = load_vocab_file(args.vocab)
    log.info("{} tokens loaded".format(len(vocab)))

    #define function to infer embedding
    def calc_emb(tokens_id, max_length):
        num = min(max_length - 2, len(tokens_id))

        # forms the request
        pad_len = max_length - num - 2
        tok_cls = [vocab['[CLS]']]
        tok_sep = [vocab['[SEP]']]
        tok_pad = [vocab['[PAD]']]

        dtype = np.int32
        inputs = {
            input_names_emb[0]: np.array([tok_cls + tokens_id[:num] + tok_sep + tok_pad * pad_len], dtype=dtype),
            input_names_emb[1]: np.array([[1]     + [1] * num       + [1]     + [0]     * pad_len], dtype=dtype),
            input_names_emb[2]: np.array([[0]     + [0] * num       + [0]     + tok_pad * pad_len], dtype=dtype),
            input_names_emb[3]: np.arange(max_length, dtype=dtype)[None, :]
        }

        # calc embedding
        ie_encoder_exec_emb = ie_encoder_exec_emb_dict[max_length]

        t_start = time.perf_counter()
        res = ie_encoder_exec_emb.infer(inputs=inputs)
        t_end = time.perf_counter()
        log.info("embedding calculated for sequence of length {} with {:0.2f} requests/sec ({:0.2} sec per request)".format(
            max_length,
            1 / (t_end - t_start),
            t_end - t_start
        ))


        res = res[output_names_model_emb[0]]
        return res.squeeze(0)

    #small class to store context as text and tokens and its embedding vector
    class ContextData:
        def __init__(self, context, c_tokens_id, c_tokens_se):
            self.context = context
            self.c_tokens_id = c_tokens_id
            self.c_tokens_se = c_tokens_se
            self.c_emb = calc_emb(self.c_tokens_id, max_length_c)

    paragraphs = get_paragraphs(args.input)
    contexts_all = []

    log.info("Indexing {} paragraphs...".format(len(paragraphs)))
    for par in paragraphs:
        c_tokens_id, c_tokens_se = text_to_tokens(par.lower(), vocab)
        if not c_tokens_id:
            continue

        # get context as string and then encode it into token id list
        # calculate number of tokens for context in each request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        if args.model_qa:
            #to make context be able to pass model_qa together with question
            c_wnd_len = max_length_qc - (max_length_q + 3)
        else:
            #to make context be able to pass model_emb without question
            c_wnd_len = max_length_c - 2

        # token num between 2 neighbours context windows
        # 1/2 means that context windows are interleaved by half
        c_stride = c_wnd_len // 2

        # init scan window
        c_s, c_e = 0, min(c_wnd_len, len(c_tokens_id))

        # iterate while context window is not empty
        while c_e > c_s:
            contexts_all.append(ContextData(par, c_tokens_id[c_s:c_e], c_tokens_se[c_s:c_e]))

            # check that context window reach the end
            if c_e == len(c_tokens_id):
                break

            # move to next window position
            c_s, c_e = c_s+c_stride, c_e+c_stride

            shift_left = max(0, c_e - len(c_tokens_id))
            c_s, c_e = c_s -shift_left, c_e-shift_left
            assert c_s >= 0, "start can be left of 0 only with window less than len but in this case we can not be here"

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

        log.info("---Stage 1---Calc question embedding and compare with {} context embeddings".format(len(contexts_all)))
        q_tokens_id, _ = text_to_tokens(question.lower(), vocab)

        q_emb = calc_emb(q_tokens_id, max_length_q)
        distances = [(np.linalg.norm(c.c_emb - q_emb, 2), c) for c in contexts_all]
        distances.sort(key=lambda x: x[0])
        keep_num = min(args.best_n, len(distances))
        distances_filtered = distances[:keep_num]

        #print short list
        print("The closest contexts to question:")
        for i, (dist, c_data) in enumerate(distances_filtered):
            print("#{}: embedding distance {} for context '{}'".format(i + 1, dist, c_data.context))

        #run model_qa if available to find exact answer to question in filtered in contexts
        if args.model_qa:

            log.info("---Stage 2---Looking for exact answers in {} contexts filtered in from {}".format(keep_num, len(distances)))
            # array of answers from each context_data
            answers = []

            for dist, c_data in distances_filtered:
                #forms the request
                tok_cls = [vocab['[CLS]']]
                tok_sep = [vocab['[SEP]']]
                tok_pad = [vocab['[PAD]']]
                req_len = len(q_tokens_id) + len(c_data.c_tokens_id) + 3
                pad_len = max_length_qc - req_len
                assert pad_len >= 0

                input_ids = tok_cls + q_tokens_id + tok_sep + c_data.c_tokens_id + tok_sep + tok_pad*pad_len
                token_type_ids = [0] * (len(q_tokens_id)+2) + [1] * (len(c_data.c_tokens_id)+1) + tok_pad * pad_len
                attention_mask = [1] * req_len + [0] * pad_len

                #create numpy inputs for IE
                inputs = {
                    input_names_qa[0]: np.array([input_ids], dtype=np.int32),
                    input_names_qa[1]: np.array([attention_mask], dtype=np.int32),
                    input_names_qa[2]: np.array([token_type_ids], dtype=np.int32),
                }
                if len(input_names_qa) > 3:
                    inputs['position_ids'] = np.arange(max_length_qc, dtype=np.int32)[None, :]

                #infer by IE
                t_start = time.perf_counter()
                res = ie_encoder_qa_exec.infer(inputs=inputs)
                t_end = time.perf_counter()
                log.info(
                    "Exact answer calculated for sequence of length {} with {:0.2f} requests/sec ({:0.2} sec per request)".format(
                        max_length_qc,
                        1 / (t_end - t_start),
                        t_end - t_start
                    ))

                #get start-end scores for context
                def get_score(name):
                    out = np.exp(res[name].reshape((max_length_qc, )))
                    return out / out.sum(axis=-1)
                score_s = get_score(output_names_qa[0])
                score_e = get_score(output_names_qa[1])

                # find product of all start-end combinations to find the best one
                c_s_idx = len(q_tokens_id) + 2 # index of first context token in tensor
                c_e_idx = max_length_qc-(1+pad_len) # index of last+1 context token in tensor
                score_mat = np.matmul(
                    score_s[c_s_idx:c_e_idx].reshape((len(c_data.c_tokens_id), 1)),
                    score_e[c_s_idx:c_e_idx].reshape((1, len(c_data.c_tokens_id)))
                )
                # reset candidates with end before start
                score_mat = np.triu(score_mat)
                # reset long candidates (>max_answer_token_num)
                score_mat = np.tril(score_mat, args.max_answer_token_num - 1)
                # find the best start-end pair
                max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
                max_score = score_mat[max_s, max_e]

                # convert to context text start-end index
                max_s = c_data.c_tokens_se[max_s][0]
                max_e = c_data.c_tokens_se[max_e][1]

                # check that answers list does not have answer yet
                # it could be because of context windows overlapping
                same = [i for i, a in enumerate(answers) if a[1] == max_s and a[2]==max_e and a[3] is c_data.context]
                if same:
                    assert len(same) == 1
                    #update exist answer record
                    a = answers[same[0]]
                    answers[same[0]] = (max(max_score, a[0]), max_s, max_e, c_data.context)
                else:
                    #add new record
                    answers.append((max_score, max_s, max_e, c_data.context))

            def mark(txt):
                return "\033[91m" + txt + "\033[0m" if args.colors else "*" + txt + "*"

            #print top 3 results
            answers.sort(key=lambda x: -x[0])
            log.info("---Stage 3---Find best 3 answers from {} results of Stage 1".format(len(answers)))
            for score, s, e, context in answers[:3]:
                print("Answer (score: {:0.2f}): {}".format(score, mark(context[s:e])))
                print(context[:s] + mark(context[s:e]) + context[e:])


if __name__ == '__main__':
    sys.exit(main() or 0)

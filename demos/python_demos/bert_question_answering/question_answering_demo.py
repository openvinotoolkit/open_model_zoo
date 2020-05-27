#!/usr/bin/env python
import logging as log
import os
import string
import sys
import time
import unicodedata
from argparse import ArgumentParser, SUPPRESS

import bs4
import numpy as np
import requests
from openvino.inference_engine import IENetwork, IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-v", "--vocab", help="Required. path to the vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=str)
    args.add_argument("--input_names",
                      help="Required. Input names for the network. For example ['input_ids','attention_mask','token_type_ids']",
                      required=True, type=str)
    args.add_argument("--output_names",
                      help="Required. Output names for the network. For example ['output_s','output_e']",
                      required=True, type=str)
    args.add_argument("--model_squad_ver", help="Optional. SQUAD version used for model fine tuning",
                      default="1.2", required=False, type=str)
    args.add_argument("-i", "--input", help="Required. Url to a page with context",
                      required=True, type=str)
    args.add_argument("-a", "--max_answer_token_num", help="Optional. Maximum number of tokens in answer",
                      default=15, required=False, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU",
                      default="CPU", type=str)
    return parser


# split word by vocab items and get tok codes
# iteratively return codes
def encode_by_voc(w, vocab):
    # remove mark and control chars
    def clean_word(w):
        # extract marks as separate chars to remove them later
        wo = ""  # accumulator for output word
        for c in unicodedata.normalize("NFD", w):
            c_cat = unicodedata.category(c)
            # remove mark nonspacing code and controls
            if c_cat != "Mn" and c_cat[0] != "C":
                wo += c
        return wo

    w = clean_word(w)
    w = w.lower()
    s, e = 0, len(w)
    while e > s:
        subword = w[s:e] if s == 0 else "##" + w[s:e]
        if subword in vocab:
            yield vocab[subword]
            s, e = e, len(w)
        else:
            e -= 1
    if s < len(w):
        yield vocab['[UNK]']


# split big text into words by spaces
# iteratively return words
def split_to_words(text):
    prev_is_sep = True  # mark initial prev as space to start word from 0 char
    for i, c in enumerate(text + " "):
        is_punc = (c in string.punctuation or unicodedata.category(c)[0] == "P")
        cur_is_sep = (c.isspace() or is_punc)
        if prev_is_sep != cur_is_sep:
            if prev_is_sep:
                start = i
            else:
                yield start, i
                del start
        if is_punc:
            yield i, i + 1
        prev_is_sep = cur_is_sep


# get the text and return list of token ids and start-end position for each id (in the original text)
def text_to_tokens(text, vocab):
    tokens_id = []
    tokens_se = []
    for s, e in split_to_words(text):
        for tok in encode_by_voc(text[s:e], vocab):
            tokens_id.append(tok)
            tokens_se.append((s, e))
    log.info("Size: {} tokens".format(len(tokens_id)))
    return tokens_id, tokens_se


# return entire sentence as start-end positions for a given answer (within the sentence).
def find_sentence_range(context, s, e):
    # find start of sentence
    for c_s in range(s, max(-1, s - 200), -1):
        if context[c_s] in "\n\.":
            c_s += 1
            break

    # find end of sentence
    for c_e in range(max(0, e - 1), min(len(context), s + 200), +1):
        if context[c_e] in "\n\.":
            break

    return c_s, c_e


# return context as one big string by given input arguments
def get_context(args):
    log.info("Get context from {}".format(args.input))
    response = requests.get(args.input)
    if response.status_code == 404:
        raise Exception('URL is invalid')
    html = bs4.BeautifulSoup(response.text, 'html.parser')
    heads = html.select("#firstHeading")
    title = heads[0].text if heads else "no title"
    paragraphs = html.select("p")
    log.info("Article title: '{}'".format(title))
    context = '\n'.join([par.text for par in paragraphs])
    log.info("Size: {} chars".format(len(context)))
    log.info("Context: \033[91m {} \033[0m".format(context))
    return context


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Initializing Inference Engine")
    ie = IECore()
    log.info("Device is {}".format(args.device))
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    # check input and output names
    input_names_model = list(ie_encoder.inputs.keys())
    output_names_model = list(ie_encoder.outputs.keys())
    input_names = eval(args.input_names)
    output_names = eval(args.output_names)

    if set(input_names_model) != set(input_names) or set(output_names_model) != set(output_names):
        log.error("Input or Output names don't match")
        log.error("     Network input->output names: {}->{}".format(input_names_model, output_names_model))
        log.error("     Expected (from the demo cmd-line) input->output names: {}->{}".format(input_names, output_names))
        raise Exception("Unexpected network input or output names")

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    # load vocabulary file for model
    log.info("Loading vocab file:\t{}".format(args.vocab))
    with open(args.vocab, "r", encoding="utf-8") as r:
        vocab = dict((t.rstrip("\n"), i) for i, t in enumerate(r.readlines()))
    log.info("{} tokens loaded".format(len(vocab)))

    # get context as a string and encode that  into token id list
    context = get_context(args)
    c_tokens_id, c_tokens_se = text_to_tokens(context, vocab)

    # loop on user's questions
    while True:
        question = input('Type question (enter to exit):')
        if not question:
            break

        q_tokens_id, _ = text_to_tokens(question, vocab)

        # maximum number of tokens that can be processed by network at once
        max_length = ie_encoder.inputs[input_names[0]].shape[1]

        # calculate number of tokens for context in each inference request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        c_wnd_len = max_length - (len(q_tokens_id) + 3)

        # token num between two neighbour context windows
        # 1/2 means that context windows are overlapped by half
        c_stride = c_wnd_len // 2

        t0 = time.time()
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
            t_start = time.time()
            # infer by IE
            res = ie_encoder_exec.infer(inputs=inputs)
            t_end = time.time()
            t_count += 1
            log.info("Sequence of length {} is processed with {:0.2f} sentence/sec ({:0.2} sec per request)".format(
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
                # update exist answer record
                a = answers[same[0]]
                answers[same[0]] = (max(max_score, a[0]), max_s, max_e)
            else:
                # add new record
                answers.append((max_score, max_s, max_e))

            # check that context window reach the end
            if c_e == len(c_tokens_id):
                break

            # move to next window position
            c_s = min(c_s + c_stride, len(c_tokens_id))
            c_e = min(c_s + c_wnd_len, len(c_tokens_id))

        t1 = time.time()
        log.info("{} requests by {} length are processed by {:0.2f}sec ({:0.2}sec per request)".format(
            t_count,
            max_length,
            t1 - t0,
            (t1 - t0) / t_count
        ))

        # print top 3 results
        answers = list(sorted(answers, key=lambda x: -x[0]))
        for score, s, e in answers[:3]:
            log.info("---answer: {:0.2f} {}".format(score, context[s:e]))
            c_s, c_e = find_sentence_range(context, s, e)
            log.info("   " + context[c_s:s] + "\033[91m" + context[s:e] + '\033[0m' + context[e:c_e])


if __name__ == '__main__':
    sys.exit(main() or 0)

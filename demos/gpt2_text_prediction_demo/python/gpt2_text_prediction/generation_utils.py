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

import json
import numpy as np


def load_vocab_file(vocab_file_name):
    with open(vocab_file_name, "r", encoding="utf-8") as content:
        return json.load(content)


def get_top_k_logits(scores, top_k):
    filter_value = -float("Inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove, fill_value=filter_value).filled()
    return filtred_scores


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum = e_x.sum(axis=-1, keepdims=True)
    return e_x / sum


def get_top_p_logits(scores, top_p):
    filter_value = -float("Inf")
    sorted_indices = np.argsort(-scores)
    sorted_logits = -np.sort(-scores)
    cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
    sorted_indices_to_remove[:, 0] = 0
    np.put_along_axis(sorted_indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)
    filtred_scores = np.ma.array(scores, mask=sorted_indices_to_remove, fill_value=filter_value).filled()
    return filtred_scores


def process_logits(input_ids, scores, eos_token_id, min_length=0):
    cur_len = input_ids.shape[-1]
    if cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def stop_criteria(input_ids, max_length, eos_token_id):
    if input_ids[0][-1] == eos_token_id:
        return True

    return input_ids.shape[-1] >= max_length

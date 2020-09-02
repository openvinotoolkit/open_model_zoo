/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on decoder_utils.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#include "decoder_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<size_t, float>> get_pruned_log_probs(
    const std::vector<float> &prob_step,
    float cutoff_prob,
    size_t cutoff_top_n,
    int log_input) {
  std::vector<std::pair<int, float>> prob_idx;
  float log_cutoff_prob = log(cutoff_prob);
  for (size_t i = 0; i < prob_step.size(); ++i) {
    prob_idx.push_back(std::pair<int, float>(i, prob_step[i]));
  }
  // pruning of vacobulary
  size_t cutoff_len = prob_step.size();
  if (log_cutoff_prob < 0.0 || cutoff_top_n < cutoff_len) {
    std::sort(
        prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int, float>);
    if (log_cutoff_prob < 0.0) {
      float cum_prob = 0.0;
      cutoff_len = 0;
      for (size_t i = 0; i < prob_idx.size(); ++i) {
        cum_prob = log_sum_exp<float>(cum_prob, log_input ? prob_idx[i].second : log(prob_idx[i].second) );
        cutoff_len += 1;
        if (cum_prob >= cutoff_prob || cutoff_len >= cutoff_top_n) break;
      }
    }else{
      cutoff_len = cutoff_top_n;
    }
    prob_idx = std::vector<std::pair<int, float>>(
        prob_idx.begin(), prob_idx.begin() + cutoff_len);
  }
  std::vector<std::pair<size_t, float>> log_prob_idx;
  for (size_t i = 0; i < cutoff_len; ++i) {
    log_prob_idx.push_back(std::pair<int, float>(
        prob_idx[i].first, log_input ? prob_idx[i].second : log(prob_idx[i].second + NUM_FLT_MIN)));
  }
  return log_prob_idx;
}


std::vector<std::pair<float, Output>> get_beam_search_result(
    const std::vector<PathTrie *> &prefixes,
    size_t beam_size) {
  // allow for the post processing
  std::vector<PathTrie *> space_prefixes;
  if (space_prefixes.empty()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      space_prefixes.push_back(prefixes[i]);
    }
  }

  std::sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
  std::vector<std::pair<float, Output>> output_vecs;
  for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i) {
    std::vector<int> output;
    std::vector<int> timesteps;
    space_prefixes[i]->get_path_vec(output, timesteps);
    Output outputs;
    outputs.tokens = output;
    outputs.timesteps = timesteps;
    std::pair<float, Output> output_pair(-space_prefixes[i]->approx_ctc,
                                               outputs);
    output_vecs.emplace_back(output_pair);
  }

  return output_vecs;
}

size_t get_utf8_str_len(const std::string &str) {
  size_t str_len = 0;
  for (char c : str) {
    str_len += ((c & 0xc0) != 0x80);
  }
  return str_len;
}

std::vector<std::string> split_utf8_str(const std::string &str) {
  std::vector<std::string> result;
  std::string out_str;

  for (char c : str) {
    if ((c & 0xc0) != 0x80)  // new UTF-8 character
    {
      if (!out_str.empty()) {
        result.push_back(out_str);
        out_str.clear();
      }
    }

    out_str.append(1, c);
  }
  result.push_back(out_str);
  return result;
}

std::vector<std::string> split_str(const std::string &s,
                                   const std::string &delim) {
  std::vector<std::string> result;
  std::size_t start = 0, delim_len = delim.size();
  while (true) {
    std::size_t end = s.find(delim, start);
    if (end == std::string::npos) {
      if (start < s.size()) {
        result.push_back(s.substr(start));
      }
      break;
    }
    if (end > start) {
      result.push_back(s.substr(start, end - start));
    }
    start = end + delim_len;
  }
  return result;
}

bool prefix_compare(const PathTrie *x, const PathTrie *y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return x->score > y->score;
  }
}

bool add_word_to_dictionary(
    const std::string &word,
    const std::unordered_map<std::string, int> &char_map,
    bool add_space,
    int space_id,
    std::vector<std::vector<int> >& int_vocabulary) {
  auto characters = split_utf8_str(word);

  std::vector<int> int_word;

  for (auto &c : characters) {
    if (c == " ") {
      int_word.push_back(space_id);
    } else {
      auto int_c = char_map.find(c);
      if (int_c != char_map.end()) {
        int_word.push_back(int_c->second);
      } else {
        return false;  // return without adding
      }
    }
  }

  if (add_space) {
    int_word.push_back(space_id);
  }

  int_vocabulary.push_back(std::move(int_word));
  return true;  // return with successful adding
}

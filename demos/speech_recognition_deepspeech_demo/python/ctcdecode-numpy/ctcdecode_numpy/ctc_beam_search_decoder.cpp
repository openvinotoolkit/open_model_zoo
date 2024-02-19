/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on ctc_beam_search_decoder.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"


CtcDecoderState::CtcDecoderState() :
  is_finalized_(false),
  next_timestep_(0),
  alphabet_(),
  space_idx_(-1),  // -2 for no space
  blank_idx_(-1),
  beam_size_(0),
  lm_scorer_(nullptr),
  cutoff_prob_(0.),
  cutoff_top_n_(0),
  candidates_trie_(),
  candidates_()  // empty candidates_ will flag uninitialized object
{}

void CtcDecoderState::init(
    const std::vector<std::string>& alphabet,
    size_t blank_idx,
    size_t beam_size,
    ScorerBase * lm_scorer) {

  is_finalized_ = false;
  next_timestep_ = 0;

  alphabet_ = alphabet;
  // assign space id
  auto it = std::find(alphabet_.begin(), alphabet_.end(), " ");
  int space_idx = int(it - alphabet_.begin());
  // if no space in vocabulary
  if (size_t(space_idx) >= alphabet_.size())
    space_idx = -2;
  space_idx_ = space_idx;

  blank_idx_ = blank_idx;
  beam_size_ = beam_size;
  lm_scorer_ = lm_scorer;

  cutoff_prob_ = 1.0;
  cutoff_top_n_ = 40;

  candidates_.clear();
  candidates_trie_.reset();
  new_sequence();
}

void CtcDecoderState::deinit() {
  lm_scorer_ = nullptr;
  alphabet_.clear();
  candidates_.clear();
  candidates_trie_.reset();
  is_finalized_ = false;
}

bool CtcDecoderState::set_config(const std::string& name, double value, bool required) {
  if (name == "beam_size")  beam_size_ = size_t(value);
  else if (name == "blank_idx")  blank_idx_ = size_t(value);
  else if (name == "cutoff_prob")  cutoff_prob_ = value;
  else if (name == "cutoff_top_n")  cutoff_top_n_ = size_t(value);
  else if (name == "next_timestep")  next_timestep_ = size_t(value);
  else {
    if (required)
      throw std::invalid_argument("CtcDecoderState::set_config(): unknown configuration parameter: " + name);
    return false;
  }
  return true;
}

double CtcDecoderState::get_config(const std::string& name, bool required, double not_found) {
  if (name == "beam_size")  return beam_size_;
  else if (name == "blank_idx")  return blank_idx_;
  else if (name == "cutoff_prob")  return cutoff_prob_;
  else if (name == "cutoff_top_n")  return cutoff_top_n_;
  else if (name == "next_timestep")  return next_timestep_;

  if (required)
    throw std::invalid_argument("CtcDecoderState::get_config(): unknown configuration parameter: " + name);
  return not_found;
}

void CtcDecoderState::new_sequence() {
  // The root of PathTrie (candidates_trie_) owns the whole trie.
  // init prefixes' root
  candidates_trie_ = std::unique_ptr<PathTrie>(new PathTrie);
  candidates_trie_->score = candidates_trie_->log_prob_b_prev = 0.0;
  candidates_.clear();
  candidates_.push_back(candidates_trie_.get());

  if (lm_scorer_ != nullptr && !lm_scorer_->is_character_based()) {
    WordPrefixSet * dict_ptr = lm_scorer_->dictionary.get();
    candidates_trie_->set_dictionary(dict_ptr);
  }

  is_finalized_ = false;
  next_timestep_ = 0;
}

void CtcDecoderState::append(
    const float * probs,
    size_t probs_frame_num,
    size_t probs_frame_stride,
    size_t probs_alph_stride,
    bool log_probs) {

  if (!candidates_trie_)
    throw std::runtime_error("CtcDecoderState::append(): call init() before append()");
  if (is_finalized_)
    throw std::runtime_error("CtcDecoderState::append(): this state was finalized, cannot append more data");

  // prefix search over time
  for (size_t time_step = 0; time_step < probs_frame_num; ++time_step) {
    const float * prob = &probs[time_step * probs_frame_stride];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (lm_scorer_ != nullptr) {
      size_t num_candidates = std::min(candidates_.size(), beam_size_);
      std::sort(
          candidates_.begin(), candidates_.begin() + num_candidates, prefix_compare);
      float blank_prob = log_probs ? prob[blank_idx_ * probs_alph_stride] : std::log(prob[blank_idx_ * probs_alph_stride]);
      min_cutoff = candidates_[num_candidates - 1]->score +
                   blank_prob - std::max(0.0, lm_scorer_->beta);
      full_beam = (num_candidates == beam_size_);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, alphabet_.size(), probs_alph_stride, cutoff_prob_, cutoff_top_n_, log_probs);
    // loop over chars
    for (size_t index = 0; index < alphabet_.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < candidates_.size() && i < beam_size_; ++i) {
        auto prefix = candidates_[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_idx_) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == size_t(prefix->character)) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get/create new prefix; returns null if the new prefix was pruned by a dictionary
        auto prefix_new = prefix->get_path_trie(c, time_step + next_timestep_, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == size_t(prefix->character) &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != size_t(prefix->character)) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (lm_scorer_ != nullptr &&
              (c == size_t(space_idx_) || lm_scorer_->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (lm_scorer_->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = lm_scorer_->make_ngram(prefix_to_score);
            score = lm_scorer_->get_log_cond_prob(ngram) * lm_scorer_->alpha;
            log_p += score;
            log_p += lm_scorer_->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary

    candidates_.clear();
    // update log probs
    candidates_trie_->iterate_to_vec(candidates_);

    // only preserve top beam_size prefixes
    if (candidates_.size() >= beam_size_) {
      std::nth_element(candidates_.begin(),
                       candidates_.begin() + beam_size_,
                       candidates_.end(),
                       prefix_compare);
      for (size_t i = beam_size_; i < candidates_.size(); ++i) {
        candidates_[i]->remove();
      }
    }
  }  // end of loop over time

  next_timestep_ += probs_frame_num;
}

void CtcDecoderState::finalize() {
  if (!candidates_trie_)
    throw std::runtime_error("CtcDecoderState::finalize(): uninitialized state");
  if (is_finalized_)
    return;
  is_finalized_ = true;

  // score the last word of each prefix that doesn't end with space
  // (NB: this may introduce duplicates, which may slightly affect quality.)
  if (lm_scorer_ != nullptr && !lm_scorer_->is_character_based()) {
    for (size_t i = 0; i < beam_size_ && i < candidates_.size(); ++i) {
      auto prefix = candidates_[i];
      if (!prefix->is_empty() && prefix->character != space_idx_) {
        float score = 0.0;
        std::vector<std::string> ngram = lm_scorer_->make_ngram(prefix);
        score = lm_scorer_->get_log_cond_prob(ngram) * lm_scorer_->alpha;
        score += lm_scorer_->beta;
        prefix->score += score;
      }
    }
  }
}

std::vector<std::pair<float, Output>> CtcDecoderState::decode(
    size_t limit_candidates,
    bool _finalize) {

  if (!candidates_trie_)
    throw std::runtime_error("CtcDecoderState::decode(): uninitialized state");
  if (_finalize)
    finalize();

  size_t num_candidates = std::min(candidates_.size(), beam_size_);
  std::sort(candidates_.begin(), candidates_.begin() + num_candidates, prefix_compare);
  if (limit_candidates > 0)
    num_candidates = std::min(num_candidates, limit_candidates);

  // Compute approximate ctc (audio model) score, without affecting the
  // return order of decoding result.
  for (size_t i = 0; i < num_candidates; ++i) {
    float approx_ctc = candidates_[i]->score;
    if (lm_scorer_ != nullptr && _finalize) {
      std::vector<int> output;
      std::vector<int> timesteps;
      candidates_[i]->get_path_vec(output, timesteps);
      auto prefix_length = output.size();
      auto words = lm_scorer_->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * lm_scorer_->beta;
      // remove language model weight
      approx_ctc -= (lm_scorer_->get_sent_log_prob(words)) * lm_scorer_->alpha;
    }
    candidates_[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(candidates_, num_candidates);
};

std::vector<std::pair<float, Output>> ctc_beam_search_decoder(
    const float * probs,
    size_t probs_frame_num,
    size_t probs_frame_stride,
    size_t probs_alph_stride,
    const std::vector<std::string>& alphabet,
    size_t beam_size,
    float cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_idx,
    bool log_probs,
    ScorerBase * lm_scorer) {

  CtcDecoderState decoder;
  decoder.init(alphabet, blank_idx, beam_size, lm_scorer);
  decoder.set_config("cutoff_prob", cutoff_prob, true);
  decoder.set_config("cutoff_top_n", cutoff_top_n, true);

  decoder.append(probs, probs_frame_num, probs_frame_stride, probs_alph_stride, log_probs);
  return decoder.decode();
}

std::vector<std::vector<std::pair<float, Output>>>
ctc_beam_search_decoder_batch(
    const float * probs,  // [probs_batch_num x probs_frame_num x alphabet.size()] array with given strides
    size_t probs_batch_num,
    const std::vector<size_t>& probs_frame_nums,  // [probs_batch_num] array
    size_t probs_batch_stride,
    size_t probs_frame_stride,
    size_t probs_alph_stride,

    const std::vector<std::string> &alphabet,
    size_t beam_size,
    size_t num_processes,
    float cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_idx,
    bool log_probs,
    ScorerBase * lm_scorer) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);

  if (probs_frame_nums.size() != probs_batch_num)
    throw std::runtime_error("ctc_beam_search_decoder_batch(): sizes of probs and probs_frame_nums arguments don't match");

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<float, Output>>>> res;
  for (size_t i = 0; i < probs_batch_num; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  &probs[i * probs_batch_stride],
                                  probs_frame_nums[i],
                                  probs_frame_stride,
                                  probs_alph_stride,
                                  alphabet,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_idx,
                                  log_probs,
                                  lm_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<float, Output>>> batch_results;
  for (size_t i = 0; i < probs_batch_num; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}

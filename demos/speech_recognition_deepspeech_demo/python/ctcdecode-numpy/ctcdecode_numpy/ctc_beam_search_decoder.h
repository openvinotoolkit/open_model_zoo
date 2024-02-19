/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in its major part on ctc_beam_search_decoder.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "scorer_base.h"
#include "output.h"
#include "path_trie.h"


class CtcDecoderState {
public:
  CtcDecoderState();
  CtcDecoderState(const CtcDecoderState&) = delete;  // because copying PathTrie is not inplemented
  CtcDecoderState(CtcDecoderState&&) = default;
  CtcDecoderState& operator=(const CtcDecoderState& src) = delete;
  CtcDecoderState& operator=(CtcDecoderState&& src) = default;

  void init(
    const std::vector<std::string>& alphabet,
    size_t blank_idx,
    size_t beam_size,
    ScorerBase * lm_scorer = nullptr  // we don't own it: the caller is responsible for destroying it at the proper time
                                      // nullptr when no scorer
  );
  void deinit();  // dispose of all resources (and forget pointer to lm_scorer, which we don't own)
  bool set_config(const std::string& name, double value, bool required = false);
  double get_config(const std::string& name, bool required = false, double not_found = -1.);

  void new_sequence();
  void append(
    const float * probs,
    size_t probs_frame_num,
    size_t probs_frame_stride,
    size_t probs_alph_stride,
    bool log_probs
  );
  void finalize();
  bool is_finalized() { return is_finalized_; }
  std::vector<std::pair<float, Output>> decode(
    size_t limit_candidates = 0,  // use 0 to return all candidates
    bool finalize = true
  );

private:
  bool is_finalized_;
  size_t next_timestep_;

  std::vector<std::string> alphabet_;
  int space_idx_;  // -2 for no space
  size_t blank_idx_;
  size_t beam_size_;
  ScorerBase * lm_scorer_;
  float cutoff_prob_;
  size_t cutoff_top_n_;

  std::unique_ptr<PathTrie> candidates_trie_;
  std::vector<PathTrie *> candidates_;  // non-owning pointers, to cache data
};

/* CTC Beam Search Decoder

 * Parameters:
 *     probs...: 2-D vector that each element is a vector of probabilities
 *            over vocabulary of one time step.
 *     alphabet: A vector of alphabet symbols.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     blank_idx: index of CTC blank (separator) symbol
 *     log_probs: true = probs contain base e log(probabilities)
 *     lm_scorer:  External LM scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<std::pair<float, Output>> ctc_beam_search_decoder(
    const float * probs,
    size_t probs_frame_num,
    size_t probs_frame_stride,
    size_t probs_alph_stride,

    const std::vector<std::string>& alphabet,
    size_t beam_size,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_idx = 0,
    bool log_probs = false,
    ScorerBase *lm_scorer = nullptr);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<float, Output>>>
ctc_beam_search_decoder_batch(
    const float * probs,  // probs_batch_num x probs_frame_num x alphabet.size() array with given strides
    size_t probs_batch_num,
    const std::vector<size_t>& probs_frame_nums,  // [probs_batch_num] array
    size_t probs_batch_stride,
    size_t probs_frame_stride,
    size_t probs_alph_stride,

    const std::vector<std::string>& alphabet,
    size_t beam_size,
    size_t num_processes,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_idx = 0,
    bool log_probs = false,
    ScorerBase *lm_scorer = nullptr);

#endif  // CTC_BEAM_SEARCH_DECODER_H_

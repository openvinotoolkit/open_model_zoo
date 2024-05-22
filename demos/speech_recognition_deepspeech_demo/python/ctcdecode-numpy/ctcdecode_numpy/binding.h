/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <vector>
#include <string>

#include "ctc_beam_search_decoder.h"


// Simple wrapper class to accept numpy arrays in SWIG in append_numpy() method
class CtcDecoderStateNumpy : public CtcDecoderState {
public:
  CtcDecoderStateNumpy() : CtcDecoderState() {}
  CtcDecoderStateNumpy(const CtcDecoderStateNumpy&) = delete;  // because copying PathTrie is not inplemented
  CtcDecoderStateNumpy(CtcDecoderStateNumpy&&) = default;
  CtcDecoderStateNumpy& operator=(const CtcDecoderStateNumpy& src) = delete;
  CtcDecoderStateNumpy& operator=(CtcDecoderStateNumpy&& src) = default;

  void append_numpy(
    const float * probs,  size_t num_frames, size_t num_classes,
    bool log_probs
  ) { append(probs, num_frames, num_classes, 1, log_probs); }
  void decode_numpy(
    int limit_candidates,  // limits candidates returned from beam search; use -1 to not limit
    bool finalize,  // normally (by default) set to "true" to finalize when getting the final decoding result

    // Output arrays (SWIG memory managed argout, malloc() allocator):
    // (here cand_size = max(beam_size, limit_candidates) )
    int ** symbols, size_t * symbols_dim,  // 1d array with concatenated symbol indices: top candidate, then second, etc.
    int ** timesteps, size_t * timesteps_dim,  // 1d array with concatenated timestamps for the symbols
    float ** scores, size_t * scores_dim,  // shape (cand_size,)
    int ** symbols_lengths, size_t * symbols_lengths_dim  // shape (cand_size,)
  );
};

// Interface to provide functionality of Parlance decoder
void batched_ctc_lm_decoder(
    const float * probs,  size_t batch_size, size_t max_frames, size_t num_classes,
    const int * seq_lens,  size_t seq_lens_dim_batch,
    const std::vector<std::string> labels,
    size_t beam_size,                 // limits candidates maintained inside beam search
    size_t max_candidates_per_batch,  // limits candidates returned from beam search
    size_t num_processes,
    float cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    bool log_probs,
    ScorerBase * lm_scorer,           // nullptr when no scorer
    // Output arrays (SWIG memory managed argout, malloc() allocator):
    // (here cand_size = max(beam_size, max_candidates_per_batch) )
    int ** symbols, size_t * symbols_dim,  // to be reshaped to (batch_size, cand_size, -1)
    int ** timesteps, size_t * timesteps_dim,  // to be reshaped to (batch_size, cand_size, -1)
    float ** scores, size_t * scores_dim,  // to be reshaped to (batch_size, cand_size)
    int ** symbols_lengths, size_t * symbols_lengths_dim);  // to be reshaped to (batch_size, cand_size)

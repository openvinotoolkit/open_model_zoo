/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on binding.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#include "binding.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

#include "scorer_yoklm.h"
#include "scorer_base.h"
#include "ctc_beam_search_decoder.h"


void CtcDecoderStateNumpy::decode_numpy(
    int limit_candidates,  // limits candidates returned from beam search; use -1 to not limit
    bool finalize,  // normally (by default) set to "true" to finalize when getting the final decoding result

    // Output arrays (SWIG memory managed argout, malloc() allocator):
    // (here cand_size = max(beam_size, limit_candidates) )
    int ** symbols, size_t * symbols_dim,  // 1d array with concatenated symbol indices: top candidate, then second, etc.
    int ** timesteps, size_t * timesteps_dim,  // 1d array with concatenated timestamps for the symbols
    float ** scores, size_t * scores_dim,  // shape (cand_size,)
    int ** symbols_lengths, size_t * symbols_lengths_dim  // shape (cand_size,)
) {
  // Get candidates
  std::vector<std::pair<float, Output>> cands = decode(limit_candidates, finalize);

  // Compute output sizes (NB: total_len can be 0)
  size_t total_len = 0;
  size_t cand_num = 0;
  for (auto&& cand : cands) {
    if (limit_candidates > 0 && cand_num >= size_t(limit_candidates))
      break;
    cand_num++;

    size_t cand_len = cand.second.tokens.size();
    if (cand_len >= std::numeric_limits<size_t>::max() - total_len)
      throw std::overflow_error("Total length of candidates returned from CtcDecoder::decode() exceeds size_t");
    total_len += cand_len;
  }

  // Allocate output arrays
  if (size_t(-1) / sizeof(**symbols) < total_len)
    throw std::runtime_error("batched_ctc_lm_decoder: size of output arg \"symbols\" exceeds size_t");
  if (size_t(-1) / sizeof(**timesteps) < total_len)
    throw std::runtime_error("batched_ctc_lm_decoder: size of output arg \"timesteps\" exceeds size_t");

  *symbols_dim = *timesteps_dim = total_len;
  *symbols = (int *)malloc(sizeof(**symbols) * *symbols_dim);
  if (*symbols == 0)
    throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() symbols");

  *timesteps = (int *)malloc(sizeof(**timesteps) * *timesteps_dim);
  if (*timesteps == 0)
    throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() timesteps");

  *scores_dim = *symbols_lengths_dim = cand_num;
  *scores = (float *)malloc(sizeof(**scores) * *scores_dim);
  if (*scores == 0)
    throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() scores");

  *symbols_lengths = (int *)malloc(sizeof(**symbols_lengths) * *symbols_lengths_dim);
  if (*symbols_lengths == 0)
    throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() symbols_lengths");

  // Copy data into the output arrays
  size_t out_idx = 0;
  for (size_t cand_idx = 0; cand_idx < cand_num; cand_idx++) {
    const std::pair<float, Output>& cand = cands[cand_idx];
    const size_t len = cand.second.tokens.size();
    if (cand.second.tokens.size() != cand.second.timesteps.size())
      throw std::logic_error("CtcDecoder::decode() returned a result with mismatching lengths of tokens and timesteps");

    (*scores)[cand_idx] = cand.first;
    (*symbols_lengths)[cand_idx] = len;
    for (size_t pos = 0; pos < len; pos++) {
      (*symbols)[out_idx] = cand.second.tokens[pos];
      (*timesteps)[out_idx] = cand.second.timesteps[pos];
      out_idx++;
    }
  }
}

size_t index_3d(size_t i1, size_t i2, size_t i3, size_t dim1, size_t dim2, size_t dim3) {
  return i3 + dim3 * (i2 + dim2 * i1);
}

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
        ScorerBase * lm_scorer,
        // Output arrays (SWIG memory managed argout, malloc() allocator):
        // (here cand_size = max(beam_size, max_candidates_per_batch) )
        int ** symbols, size_t * symbols_dim,  // to be reshaped to (batch_size, cand_size, -1)
        int ** timesteps, size_t * timesteps_dim,  // to be reshaped to (batch_size, cand_size, -1)
        float ** scores, size_t * scores_dim,  // to be reshaped to (batch_size, cand_size)
        int ** symbols_lengths, size_t * symbols_lengths_dim)  // to be reshaped to (batch_size, cand_size)
{
    if (seq_lens_dim_batch != batch_size)
        throw std::runtime_error("beam_decode: probs and seq_lens batch sizes differ");
    if (num_classes != labels.size())
        throw std::runtime_error("beam_decode: the last dimension of probs must be the size of the alphabet");
    if (max_candidates_per_batch > beam_size)
        max_candidates_per_batch = beam_size;
    if (max_candidates_per_batch < 1)
        throw std::runtime_error("batched_ctc_lm_decoder: max_candidates_per_batch must be at least 1");

    std::vector<size_t> seq_lens_vec;
    seq_lens_vec.resize(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
      if (seq_lens[b] < 0)
          throw std::runtime_error("beam_decode: negative integer in seq_lens[]");
      if (size_t(seq_lens[b]) > max_frames)
          throw std::runtime_error("beam_decode: value in seq_lens[] exceeds probs[] array size");
      seq_lens_vec[b] = seq_lens[b];
    }

    std::vector<std::vector<std::pair<float, Output> > > batch_results =
        ctc_beam_search_decoder_batch(probs, batch_size, seq_lens_vec, max_frames*num_classes, num_classes, 1,
            labels, beam_size, num_processes,
            cutoff_prob, cutoff_top_n, blank_id, log_probs, lm_scorer);

    if (batch_results.size() != batch_size)
        throw std::runtime_error("batched_ctc_lm_decoder: internal error: output batch size differs from input batch size");

    size_t max_len = 1;
    for (auto&& result_batch_entry : batch_results) {
        size_t candidate_idx = 0;
        for (auto&& result_candidate : result_batch_entry) {
            if (candidate_idx++ >= max_candidates_per_batch)
                break;
            size_t len = result_candidate.second.tokens.size();
            if (max_len < len)
                max_len = len;
        }
    }

    if ((size_t)-1 / sizeof(**symbols) / batch_size / max_candidates_per_batch / max_len == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: dimension of output arg \"symbols\" exceeds size_t");
    if ((size_t)-1 / sizeof(**timesteps) / batch_size / max_candidates_per_batch / max_len == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: dimension of output arg \"timesteps\" exceeds size_t");

    *symbols_dim = *timesteps_dim = batch_size * max_candidates_per_batch * max_len;
    *symbols = (int *)malloc(sizeof(**symbols) * *symbols_dim);
    if (*symbols == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() symbols");

    *timesteps = (int *)malloc(sizeof(**timesteps) * *timesteps_dim);
    if (*timesteps == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() timesteps");

    *scores_dim = *symbols_lengths_dim = batch_size * max_candidates_per_batch;
    *scores = (float *)malloc(sizeof(**scores) * *scores_dim);
    if (*scores == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() scores");

    *symbols_lengths = (int *)malloc(sizeof(**symbols_lengths) * *symbols_lengths_dim);
    if (*symbols_lengths == 0)
        throw std::runtime_error("batched_ctc_lm_decoder: cannot malloc() symbols_lengths");

    for (size_t b = 0; b < batch_results.size(); b++) {
        std::vector<std::pair<float, Output> >& results = batch_results[b];
        size_t p = 0;
        for (; p < results.size() && p < max_candidates_per_batch; p++) {
            const size_t index_bp = b * max_candidates_per_batch + p;
            std::pair<float, Output>& n_path_result = results[p];
            Output& output = n_path_result.second;
            for (size_t t = 0; t < output.tokens.size(); t++) {
                (*symbols)[index_bp * max_len + t] = output.tokens[t]; // fill output symbols
                (*timesteps)[index_bp * max_len + t] = output.timesteps[t];
            }
            (*scores)[index_bp] = n_path_result.first;  // scores are -log(p), so lower = better
            (*symbols_lengths)[index_bp] = output.tokens.size();
        }
        for (; p < max_candidates_per_batch; p++) {
            const size_t index_bp = b * max_candidates_per_batch + p;
            // fill the absent candidates with infitite scores and no symbols
            (*scores)[index_bp] = std::numeric_limits<float>::infinity();
            (*symbols_lengths)[index_bp] = 0;
        }
    }
}

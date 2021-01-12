/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on binding.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

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

size_t index_3d(size_t i1, size_t i2, size_t i3, size_t dim1, size_t dim2, size_t dim3) {
  return i3 + dim3 * (i2 + dim2 * i1);
}

void numpy_beam_decode(
        const float * probs,  size_t batch_size, size_t max_frames, size_t num_classes,
        const int * seq_lens,  size_t seq_lens_dim_batch,
        const std::vector<std::string> labels,
        size_t beam_size,                 // limits candidates maintained inside beam search
        size_t max_candidates_per_batch,  // limits candidates returned from beam search
        size_t num_processes,
        float cutoff_prob,
        size_t cutoff_top_n,
        size_t blank_id,
        bool log_input,
        void *scorer,
        // Output arrays (SWIG memory managed argout, malloc() allocator):
        // (here cand_size = max(beam_size, max_candidates_per_batch) )
        int ** tokens, size_t * tokens_dim,  // to be reshaped to (batch_size, cand_size, -1)
        int ** timesteps, size_t * timesteps_dim,  // to be reshaped to (batch_size, cand_size, -1)
        float ** scores, size_t * scores_dim,  // to be reshaped to (batch_size, cand_size)
        int ** tokens_lengths, size_t * tokens_lengths_dim)  // to be reshaped to (batch_size, cand_size)
{
    ScorerBase *ext_scorer = NULL;
    if (scorer != NULL) {
        ext_scorer = static_cast<ScorerBase *>(scorer);
    }

    if (seq_lens_dim_batch != batch_size)
        throw std::runtime_error("beam_decode: probs and seq_lens batch sizes differ");
    if (max_candidates_per_batch > beam_size)
        max_candidates_per_batch = beam_size;
    if (max_candidates_per_batch < 1)
        throw std::runtime_error("numpy_beam_decode: max_candidates_per_batch must be at least 1");

    std::vector<std::vector<std::vector<float> > > probs_vec;
    probs_vec.reserve(batch_size);

    for (size_t b = 0; b < batch_size; b++) {
        // ensure that an erroneous seq_len doesn't make us try to access memory we shouldn't
        if (seq_lens[b] < 0)
            throw std::runtime_error("beam_decode: negative integer in seq_lens[]");
        size_t seq_len = std::min((size_t)(seq_lens[b]), max_frames);
        std::vector<std::vector<float> > probs_one_batch(seq_len, std::vector<float>(num_classes));
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t n = 0; n < num_classes; n++) {
                float val = probs[index_3d(b,t,n, batch_size, max_frames, num_classes)];
                probs_one_batch[t][n] = val;
            }
        }
        probs_vec.emplace_back(std::move(probs_one_batch));
    }

    std::vector<std::vector<std::pair<float, Output> > > batch_results =
        ctc_beam_search_decoder_batch(probs_vec, labels, beam_size, num_processes,
            cutoff_prob, cutoff_top_n, blank_id, log_input, ext_scorer);

    if (batch_results.size() != batch_size)
        throw std::runtime_error("numpy_beam_decode: internal error: output batch size differs from input batch size");

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

    if ((size_t)-1 / sizeof(**tokens) / batch_size / max_candidates_per_batch / max_len == 0)
        throw std::runtime_error("numpy_beam_decode: dimension of output arg \"tokens\" exceeds size_t");
    if ((size_t)-1 / sizeof(**timesteps) / batch_size / max_candidates_per_batch / max_len == 0)
        throw std::runtime_error("numpy_beam_decode: dimension of output arg \"timesteps\" exceeds size_t");

    *tokens_dim = *timesteps_dim = batch_size * max_candidates_per_batch * max_len;
    *tokens = (int *)malloc(sizeof(**tokens) * *tokens_dim);
    if (*tokens == 0)
        throw std::runtime_error("numpy_beam_decode: cannot malloc() tokens");

    *timesteps = (int *)malloc(sizeof(**timesteps) * *timesteps_dim);
    if (*timesteps == 0)
        throw std::runtime_error("numpy_beam_decode: cannot malloc() timesteps");

    *scores_dim = *tokens_lengths_dim = batch_size * max_candidates_per_batch;
    *scores = (float *)malloc(sizeof(**scores) * *scores_dim);
    if (*scores == 0)
        throw std::runtime_error("numpy_beam_decode: cannot malloc() scores");

    *tokens_lengths = (int *)malloc(sizeof(**tokens_lengths) * *tokens_lengths_dim);
    if (*tokens_lengths == 0)
        throw std::runtime_error("numpy_beam_decode: cannot malloc() tokens_lengths");

    for (size_t b = 0; b < batch_results.size(); b++) {
        std::vector<std::pair<float, Output> >& results = batch_results[b];
        size_t p = 0;
        for (; p < results.size() && p < max_candidates_per_batch; p++) {
            const size_t index_bp = b * max_candidates_per_batch + p;
            std::pair<float, Output>& n_path_result = results[p];
            Output& output = n_path_result.second;
            for (size_t t = 0; t < output.tokens.size(); t++) {
                (*tokens)[index_bp * max_len + t] = output.tokens[t]; // fill output tokens
                (*timesteps)[index_bp * max_len + t] = output.timesteps[t];
            }
            (*scores)[index_bp] = n_path_result.first;  // scores are -log(p), so lower = better
            (*tokens_lengths)[index_bp] = output.tokens.size();
        }
        for (; p < max_candidates_per_batch; p++) {
            const size_t index_bp = b * max_candidates_per_batch + p;
            // fill the absent candidates with infitite scores and no tokens
            (*scores)[index_bp] = std::numeric_limits<float>::infinity();
            (*tokens_lengths)[index_bp] = 0;
        }
    }
}

void numpy_beam_decode_no_lm(
        const float * probs,  size_t batch_size, size_t max_frames, size_t num_classes,
        const int * seq_lens,  size_t seq_lens_dim_batch,
        const std::vector<std::string> labels,
        size_t beam_size,                 // limits candidates maintained inside beam search
        size_t max_candidates_per_batch,  // limits candidates returned from beam search
        size_t num_processes,
        float cutoff_prob,
        size_t cutoff_top_n,
        size_t blank_id,
        bool log_input,
        // Output arrays (SWIG memory managed argout, malloc() allocator):
        int ** tokens, size_t * tokens_dim,  // to be reshaped to (batch_size, beam_size, -1)
        int ** timesteps, size_t * timesteps_dim,  // to be reshaped to (batch_size, beam_size, -1)
        float ** scores, size_t * scores_dim,  // to be reshaped to (batch_size, beam_size)
        int ** tokens_lengths, size_t * tokens_lengths_dim)  // to be reshaped to (batch_size, beam_size)
{
    numpy_beam_decode(
        probs,  batch_size, max_frames, num_classes,
        seq_lens,  seq_lens_dim_batch,
        labels,
        beam_size,
        max_candidates_per_batch,
        num_processes,
        cutoff_prob,
        cutoff_top_n,
        blank_id,
        log_input,
        0,
        tokens, tokens_dim,
        timesteps, timesteps_dim,
        scores, scores_dim,
        tokens_lengths, tokens_lengths_dim
    );
}


void* create_scorer_yoklm(
        double alpha,
        double beta,
        const std::string& lm_path,
        const std::vector<std::string>& labels)
{
    ScorerBase* scorer = new ScorerYoklm(alpha, beta, lm_path, labels);
    return static_cast<void*>(scorer);
}

void delete_scorer(void* scorer) {
    delete static_cast<ScorerBase*>(scorer);
}

int is_character_based(void *scorer){
    ScorerBase *ext_scorer  = static_cast<ScorerBase *>(scorer);
    return ext_scorer->is_character_based();
}

size_t get_max_order(void *scorer){
    ScorerBase *ext_scorer  = static_cast<ScorerBase *>(scorer);
    return ext_scorer->get_max_order();
}

size_t get_dict_size(void *scorer){
    ScorerBase *ext_scorer  = static_cast<ScorerBase *>(scorer);
    return ext_scorer->get_dict_size();
}

void reset_params(void *scorer, double alpha, double beta){
    ScorerBase *ext_scorer  = static_cast<ScorerBase *>(scorer);
    ext_scorer->reset_params(alpha, beta);
}

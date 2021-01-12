/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <vector>
#include <string>

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
        int ** tokens, size_t * tokens_dim,  // to be reshaped to (batch_size, beam_size, -1)
        int ** timesteps, size_t * timesteps_dim,  // to be reshaped to (batch_size, beam_size, -1)
        float ** scores, size_t * scores_dim,  // to be reshaped to (batch_size, beam_size)
        int ** tokens_lengths, size_t * tokens_lengths_dim);  // to be reshaped to (batch_size, beam_size)

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
        int ** tokens_lengths, size_t * tokens_lengths_dim);  // to be reshaped to (batch_size, beam_size)

void* create_scorer_yoklm(
        double alpha,
        double beta,
        const std::string& lm_path,
        const std::vector<std::string>& labels);

void delete_scorer(void* scorer);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);

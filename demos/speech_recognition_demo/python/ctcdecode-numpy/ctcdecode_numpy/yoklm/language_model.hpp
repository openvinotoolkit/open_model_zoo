/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#ifndef YOKLM_LANGUAGE_MODEL_HPP
#define YOKLM_LANGUAGE_MODEL_HPP

#include <vector>
#include <string>
#include <utility>

#include "word_index.hpp"
#include "memory_section.hpp"


namespace yoklm {

struct UnigramNodeFormat {
  float prob;  // log10-prob for this node
  float backoff;  // log10-backoff for this node
  uint64_t start_index;  // starting index of the subtrie in the next trie layer
};

struct MediumLayer {
  int bhiksha_total_bits;
  int bhiksha_low_bits;  // TODO: get rid of data duplication here
  int backoff_bits;  // TODO: get rid of data duplication here

  size_t bhiksha_highs_count;  // number of elements in bhiksha_highs
  MemorySectionArray<uint64_t> bhiksha_highs;
  MemorySectionBitArray bit_array;  // contains bit fields: word_index, backoff, prob, bhiksha_low_bits

  BitField word_field;
  BitField backoff_field;
  BitField prob_field;
  BitField bhiksha_low_field;
};

struct LmConfig {
  size_t order;  // the value of n in n-gram
  std::vector<uint64_t> ngram_counts;  // ngram_count[k-1] = number of k-grams, vector length = order
  int prob_bits;  // size of quantized representation of log-probability values; guaranteed to be in [0,24]
  int backoff_bits;  // size of quantized representation of log-backoff values; guaranteed to be in [0,24]
  std::vector<MemorySectionArray<float> > prob_quant_tables;  // [k-2] for k-grams, k=2...n
  std::vector<MemorySectionArray<float> > backoff_quant_tables;  // [k-2] for k-grams, k=2...(n-1)
  MemorySectionArray<UnigramNodeFormat> unigram_layer;
  std::vector<MediumLayer> medium_layers;
  //MediumLayer leaves_layer;
};

struct LmState {
  LmState() {}
  explicit LmState(int order) {
    backoffs.reserve(order - 1);
    context_words.reserve(order);
  }

  // Backoffs in reverse order: backoffs[0] for 1-gram, backoffs[1] for 2-gram, etc
  std::vector<float> backoffs;
  // Up to (order-1) last words in reverse order
  std::vector<WordIndex> context_words;
};

class LanguageModel {
  public:
    LanguageModel() : config_() {}
    void load(LmConfig config) { config_ = config; }

    //float log10_p_cond(const std::vector<WordIndex>& words) const;
    float log10_p_cond(WordIndex new_word, LmState& state) const;

    size_t order() const { return config_.order; };
    uint64_t num_words() const { return config_.ngram_counts[0]; };

  private:
    LmConfig config_;

    // Accepts n-gram in state.words (as const, reverse order: the last word in state.words[0])
    // Returns:
    //   * Return value (float) = raw p-value for the longest n-gram present in the LM
    //   * state.backoff.size(): the length of the longest n-gram present in the LM
    //   * state.backoff[]: state.backoff[k] if backoff for (k+1)-gram postfix
    float find_ngram(LmState& state) const;
    //std::pair<uint64_t, uint64_t> bhiksha_lookup(const MemorySectionArray<uint64_t>& bhiksha_highs, uint64_t index) const;
};

} // namespace yoklm


#endif // YOKLM_LANGUAGE_MODEL_HPP

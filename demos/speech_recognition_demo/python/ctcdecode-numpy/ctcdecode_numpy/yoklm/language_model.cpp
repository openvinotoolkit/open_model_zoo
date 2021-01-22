/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <iostream>
#include <algorithm>

#include "sorted_search.hpp"
#include "language_model.hpp"


namespace yoklm {

float LanguageModel::log10_p_cond(WordIndex new_word, LmState& state) const {
  LmState new_state(config_.order);
  std::vector<WordIndex>& words = new_state.context_words;  // a shorthand

  words = std::move(state.context_words);
  if (words.size() > config_.order - 1)  // just in case
    words.resize(config_.order - 1);
  words.insert(words.begin(), new_word);

  // First fetch log10-p without backoff, then add backoff
  float p = find_ngram(new_state);
  // The length of the longest sententce postfix present in the LM
  size_t ngram_length = new_state.backoffs.size();
  for (size_t k = ngram_length; k <= state.backoffs.size(); k++)
    p += state.backoffs[k-1];

  if (words.size() > config_.order - 1)
    words.resize(config_.order - 1);
  state = std::move(new_state);
  return p;
}

//   Preconditions:
// bhiksha_highs is non-decreasing array, bhiksha_highs[0] = 0
// bhiksha_highs_count > 0 is the number of elements in bhiksha_highs
// 2*bhiksha_highs_count fits into uint64_t (precondition for binary_search)
// (index + 1) fits into uint64_t
//   Returns:
// pair (l,r), where
//   l is the highest index sush that bhiksha_highs[l] <= index
//   r is the highest index sush that bhiksha_highs[r] <= index + 1
std::pair<uint64_t, uint64_t> bhiksha_lookup(
    const MemorySectionArray<uint64_t>& bhiksha_highs,
    size_t bhiksha_highs_count,
    uint64_t index)
{
  // Find l = the last index with bhiksha_highs[l] <= index
  const uint64_t l = binary_search<MemorySectionArray<uint64_t>, uint64_t, uint64_t>(
    bhiksha_highs,
    0, bhiksha_highs_count,  // kenlm files have bhiksha_highs[0] = 0
    index
  );

  // Find r = the last index with bhiksha_highs[r] <= index+1
  uint64_t r = l + 1;
  while (r < bhiksha_highs_count && bhiksha_highs[r] == (index + 1))
    r++;
  r--;

  return std::make_pair(l, r);
}

float LanguageModel::find_ngram(LmState& words_backoffs) const {
  const std::vector<WordIndex>& words = words_backoffs.context_words;
  std::vector<float>& backoffs = words_backoffs.backoffs;

  backoffs.clear();

  float p;
  uint64_t l, r;

  // 1-gram
  p = config_.unigram_layer[words[0]].prob;
  backoffs.push_back(config_.unigram_layer[words[0]].backoff);
  l = config_.unigram_layer[words[0]].start_index;
  r = config_.unigram_layer[words[0] + 1].start_index;

  // Medium trie layers: 2-gram .. (n-1)-gram, and including n-gram.
  const WordIndex not_found = (WordIndex)(-1);
  for (size_t k = 2; k <= config_.order && k < words.size() + 1 && l < r; k++) {
    const WordIndex word = words[k-1];
    const MediumLayer& layer = config_.medium_layers[k-2];

    const uint64_t index = secant_search<MemorySectionBitArray, WordIndex, uint64_t>(
      layer.bit_array,  // array
      l, r,  // l, r
      0, config_.ngram_counts[0],  // plv, rv
      not_found,  // not_found
      word  // value
    );

    if (index == not_found)
      break;
    p = config_.prob_quant_tables[k-2][layer.bit_array(index, layer.prob_field)];
    if (k >= config_.order) {
      // No backoff in full n-grams.
      // But we use backoffs.size() to indicate the length of the longest postfix k-gram present in the LM
      backoffs.push_back(0);
      break;
    }

    backoffs.push_back(config_.backoff_quant_tables[k-2][layer.bit_array(index, layer.backoff_field)]);

    // Fetch index range in the next layer
    const uint64_t next_l_low = layer.bit_array(index, layer.bhiksha_low_field);
    const uint64_t next_r_low = layer.bit_array(index + 1, layer.bhiksha_low_field);
    const std::pair<uint64_t, uint64_t> next_high_lr =
      bhiksha_lookup(layer.bhiksha_highs, layer.bhiksha_highs_count, index);
    l = (next_high_lr.first << layer.bhiksha_low_bits) + next_l_low;
    r = (next_high_lr.second << layer.bhiksha_low_bits) + next_r_low;

    if (l >= r)
      break;
  }

  return p;
}


} // namespace yoklm

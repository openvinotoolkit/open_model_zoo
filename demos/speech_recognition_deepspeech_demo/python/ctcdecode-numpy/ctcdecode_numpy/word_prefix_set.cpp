/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include "word_prefix_set.h"

#include <vector>
#include <algorithm>

typedef std::vector<int> IntWord;

bool lex_less(const IntWord* a, const IntWord* b) {
  return *a < *b;
}

bool int_word_equal(const IntWord* a, const IntWord* b) {
  return *a == *b;
}

size_t WordPrefixSet::add_words(const std::vector<std::vector<int> >& words) {
  // Copy pointers to words
  std::vector<const IntWord*> word_ptrs;
  word_ptrs.reserve(words.size());
  for (const auto& word : words)
    word_ptrs.push_back(&word);

  // Sort pointers to word lexicographically
  std::sort(word_ptrs.begin(), word_ptrs.end(), lex_less);

  // Deduplicate
  auto last_it = std::unique(word_ptrs.begin(), word_ptrs.end(), int_word_equal);
  word_ptrs.erase(last_it, word_ptrs.end());

  trie_chars_.clear();
  trie_starts_.clear();
  trie_weights_.clear();

  // Go through the sorted list, and update children lists at every layer of trie.
  // The new child may only be the last child in each list.
  for (auto word : word_ptrs) {
    size_t char_index;
    // Skip the prefixes of the new word that are already present in the trie
    for (char_index = 0; char_index < word->size() && char_index < trie_chars_.size(); char_index++) {
      if (trie_chars_[char_index].back() != (*word)[char_index])
        break;
      // For our particular arc weight (char_index + 1 == word->size()), the following
      // line would never change trie_weights_, since we process words in lexicographic order.
      // To use this algo with other arc weights, the following line needs to be uncommented:
      //trie_weights_[char_index].back() |= (char_index + 1 == word->size());
    }
    // Add the new prefixes
    for (; char_index < word->size() && char_index + 1 < trie_chars_.size(); char_index++) {
      trie_chars_[char_index].push_back((*word)[char_index]);
      trie_starts_[char_index].push_back(trie_chars_[char_index + 1].size());
      trie_weights_[char_index].push_back(char_index + 1 == word->size());
    }
    for (; char_index < word->size() && char_index + 1 == trie_chars_.size(); char_index++) {
      trie_chars_[char_index].push_back((*word)[char_index]);
      trie_starts_[char_index].push_back(0);
      trie_weights_[char_index].push_back(char_index + 1 == word->size());
    }
    for (; char_index < word->size(); char_index++) {
      trie_chars_.emplace_back(1, (*word)[char_index]);
      trie_starts_.emplace_back(1, 0);
      trie_weights_.emplace_back(1, char_index + 1 == word->size());
    }
  }

  // Finalize trie_starts_ by appending the size of the next layer
  size_t prefix_length;
  for (prefix_length = 0; prefix_length + 1 < trie_chars_.size(); prefix_length++)
    trie_starts_[prefix_length].push_back(trie_chars_[prefix_length + 1].size());
  for (; prefix_length < trie_starts_.size(); prefix_length++)
    trie_starts_[prefix_length].push_back(0);

  return word_ptrs.size();
}

WordPrefixSetState WordPrefixSet::empty_state() {
  WordPrefixSetState empty;
  empty.length = 0;
  empty.start = 0;
  empty.end = (trie_chars_.size() != 0) ? trie_chars_[0].size() : 0;
  empty.weight = false;
  return empty;
}

bool WordPrefixSet::append_character(int character, WordPrefixSetState& state) {
  for (size_t next_index = state.start; next_index < state.end; next_index++)
    if (trie_chars_[state.length][next_index] == character) {
      state.start = trie_starts_[state.length][next_index];
      state.end = trie_starts_[state.length][next_index + 1];
      state.weight |= trie_weights_[state.length][next_index];
      state.length++;
      return true;
    }
  state.length = 0;
  state.start = 0;
  state.end = (trie_chars_.size() != 0) ? trie_chars_[0].size() : 0;
  state.weight = false;
  return false;
}

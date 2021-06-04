/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

// This data structure stores the set of all prefixes of vocabulary words,
// and supports quick checking of inclusion after appending a character
// to a prefix.

#ifndef WORD_PREFIX_SET_H
#define WORD_PREFIX_SET_H

#include <vector>
#include <cstdint>

struct WordPrefixSetState {
  // Length of the current prefix
  size_t length;
  // Starting and ending indices in the arrays for the next trie layer, that is in trie_*[length] if it exists
  // When trie_*[length] do not exist, we have start==end==0.
  size_t start, end;
  // Weight of the current prefix defines as the logical OR of the weights of all its trie arcs.
  // "weight" must be public.
  bool weight;
};

class WordPrefixSet {
public:
  WordPrefixSet() = default;
  WordPrefixSet(const WordPrefixSet&) = delete;
  WordPrefixSet& operator=(const WordPrefixSet&) = default;

  // Fill (replace) prefix set with all prefixes of the provided words.
  // Return the number of unique full words.
  size_t add_words(const std::vector<std::vector<int> >& words);

  // Get a new state corresponding to an empty string
  WordPrefixSetState empty_state();

  // Append a character, and update state in place.
  // If the new state would correspond to a non-existent prefix, return false
  // and reset to an empty state.
  // Return true if the new prefix exists (that is, it is a prefix of a word in
  // the vocabulary).
  bool append_character(int character, WordPrefixSetState& state);

private:
  // Each trie node is a triple (len, start, end), that is accessible from the root (0,0,len(trie_chars_[0])).
  // Node's children arc labels are trie_chars_[len][index] for index in [begin, end).
  // If trie_chars_[len] exists, then it has at least one element.
  std::vector<std::vector<int>> trie_chars_;
  // Node's children nodes ranges (begin,end) = (trie_starts_[len][index], trie_starts_[len][index+1]).
  // trie_starts_[len] is always one element longer than trie_chars_[len].
  std::vector<std::vector<size_t>> trie_starts_;
  // Weights of the children arcs. Weight of the prefix is logical OR of its arcs.
  // It is ad-hoc initialized to true for the final arcs in the words.
  std::vector<std::vector<bool>> trie_weights_;
};

#endif  // WORD_PREFIX_SET_H

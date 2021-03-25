/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on path_trie.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "word_prefix_set.h"

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
  PathTrie();
  ~PathTrie();

  // get new prefix after appending new char
  PathTrie* get_path_trie(int new_char, int new_timestep, float log_prob_c, bool reset = true);

  // get the prefix in index from root to current node
  PathTrie* get_path_vec(std::vector<int>& output, std::vector<int>& timesteps);

  // get the prefix in index from some stop node to current node
  PathTrie* get_path_vec(std::vector<int>& output,
                         std::vector<int>& timesteps,
                         int stop,
                         size_t max_steps = std::numeric_limits<size_t>::max());

  // update log probs
  void iterate_to_vec(std::vector<PathTrie*>& output);

  // set word prefix dictionary
  void set_dictionary(WordPrefixSet* dictionary);

  bool is_empty() { return ROOT_ == character; }

  // remove current path from root
  void remove();

  float log_prob_b_prev;
  float log_prob_nb_prev;
  float log_prob_b_cur;
  float log_prob_nb_cur;
  float log_prob_c;
  float score;
  float approx_ctc;
  int character;
  int timestep;
  PathTrie* parent;

private:
  int ROOT_;
  bool exists_;
  bool has_dictionary_;

  std::vector<std::pair<int, PathTrie*>> children_;

  // pointer to word prefix dictionary
  WordPrefixSet* dictionary_;
  WordPrefixSetState dictionary_state_;
};

#endif  // PATH_TRIE_H

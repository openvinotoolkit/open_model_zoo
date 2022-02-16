/*********************************************************************
* Copyright (c) 2020-2021 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on decoder_utils.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef DECODER_UTILS_H_
#define DECODER_UTILS_H_

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <iostream>
#include <algorithm>  // std::max
#include <stdexcept>
#include <unordered_map>

#include "path_trie.h"
#include "output.h"


const float NUM_FLT_INF  = std::numeric_limits<float>::max();
const float NUM_FLT_MIN  = std::numeric_limits<float>::min();
const float NUM_FLT_LOGE = 0.4342944819f;

// inline function for validation check
inline void check(
    bool x, const char *expr, const char *file, int line, const char *err) {
  if (!x) {
    std::cout << "[" << file << ":" << line << "] ";
    throw std::logic_error("\"" + std::string(expr) + "\" check failed. " + std::string(err));
  }
}

#define VALID_CHECK(x, info) \
  check(static_cast<bool>(x), #x, __FILE__, __LINE__, info)
#define VALID_CHECK_EQ(x, y, info) VALID_CHECK((x) == (y), info)
#define VALID_CHECK_GT(x, y, info) VALID_CHECK((x) > (y), info)
#define VALID_CHECK_LT(x, y, info) VALID_CHECK((x) < (y), info)


// Function template for comparing two pairs
template <typename T1, typename T2>
bool pair_comp_first_rev(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) {
  return a.first > b.first;
}

// Function template for comparing two pairs
template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> &a,
                          const std::pair<T1, T2> &b) {
  return a.second > b.second;
}

// Return the sum of two probabilities in log scale
template <typename T>
T log_sum_exp(const T &x, const T &y) {
  static T num_min = -std::numeric_limits<T>::max();
  if (x <= num_min) return y;
  if (y <= num_min) return x;
  T xmax = std::max(x, y);
  return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

// Get pruned probability vector for each time step's beam search
std::vector<std::pair<size_t, float>> get_pruned_log_probs(
    const float * prob_step,
    size_t alphabet_len,
    size_t prob_stride,
    float cutoff_prob,
    size_t cutoff_top_n,
    bool log_probs);

// Get beam search result from prefixes in trie tree
std::vector<std::pair<float, Output>> get_beam_search_result(
    const std::vector<PathTrie *> &prefixes,
    size_t beam_size);

// Functor for prefix comparison
bool prefix_compare(const PathTrie *x, const PathTrie *y);

/* Get length of utf8 encoding string
 * See: http://stackoverflow.com/a/4063229
 */
size_t get_utf8_str_len(const std::string &str);

/* Split a string into a list of strings on a given string
 * delimiter. NB: delimiters on beginning / end of string are
 * trimmed. Eg, "FooBarFoo" split on "Foo" returns ["Bar"].
 */
std::vector<std::string> split_str(const std::string &s,
                                   const std::string &delim);

/* Splits string into vector of strings representing
 * UTF-8 characters (not same as chars)
 */
std::vector<std::string> split_utf8_str(const std::string &str);

// Add a word in string to dictionary
bool add_word_to_dictionary(
    const std::string &word,
    const std::unordered_map<std::string, int> &char_map,
    bool add_space,
    int space_id,
    std::vector<std::vector<int> >& int_vocabulary);
#endif  // DECODER_UTILS_H

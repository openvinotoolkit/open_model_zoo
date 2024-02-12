/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on scorer.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#include "scorer_base.h"

#include <cassert>

#include "decoder_utils.h"

ScorerBase::ScorerBase(double alpha,
                       double beta)
      : alpha(alpha), beta(beta), dictionary(nullptr),
        is_character_based_(true), max_order_(0),
        dict_size_(0), space_id_(-1) {
  // Derived classes must call setup() from derived class constructor
  // since setup() calls virtual method load_lm():
  //   setup(lm_path, vocab_list);
}

ScorerBase::~ScorerBase() {}

void ScorerBase::setup(const std::string& lm_path,
                       const std::vector<std::string>& vocab_list) {
  // load language model
  load_lm(lm_path);
  // set char map for scorer
  set_char_map(vocab_list);
  // fill word prefix dictionary
  if (!is_character_based()) {
    fill_dictionary(true);
  }
}

double ScorerBase::get_sent_log_prob(const std::vector<std::string>& words) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence);
}

double ScorerBase::get_log_prob(const std::vector<std::string>& words) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    score += get_log_cond_prob(ngram);
  }
  return score;
}

void ScorerBase::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::string ScorerBase::vec2str(const std::vector<int>& input) {
  std::string word;
  for (auto ind : input) {
    word += char_list_[ind];
  }
  return word;
}

std::vector<std::string> ScorerBase::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};

  std::string s = vec2str(labels);
  std::vector<std::string> words;
  if (is_character_based_) {
    words = split_utf8_str(s);
  } else {
    words = split_str(s, " ");
  }
  return words;
}

void ScorerBase::set_char_map(const std::vector<std::string>& char_list) {
  char_list_ = char_list;
  char_map_.clear();

  for (size_t i = 0; i < char_list_.size(); i++) {
    if (char_list_[i] == " ") {
      space_id_ = i;
    }
    // The original implementation avoided 0, we keep this behavior for now for simplicity:
    //   "The initial state of FST is state 0, hence the index of chars in
    //   the FST should start from 1 to avoid the conflict with the initial
    //   state, otherwise wrong decoding results would be given."
    char_map_[char_list_[i]] = i + 1;
  }
}

std::vector<std::string> ScorerBase::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;

  for (size_t order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;
    std::vector<int> prefix_steps;

    if (is_character_based_) {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, -1, 1);
      current_node = new_node;
    } else {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, space_id_);
      current_node = new_node->parent;  // Skipping spaces
    }

    // reconstruct word
    std::string word = vec2str(prefix_vec);
    ngram.push_back(word);

    if (new_node->character == -1) {
      // No more spaces, but still need order
      for (size_t i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void ScorerBase::fill_dictionary(bool add_space) {
  // For each unigram convert to ints and store
  std::vector<std::vector<int> > int_vocabulary;
  for (const auto& word : vocabulary_) {
    add_word_to_dictionary(word, char_map_, add_space, space_id_ + 1, int_vocabulary);
  }

  // Add the converted vocabulary to WordPrefixSet
  this->dictionary.reset(new WordPrefixSet);
  dict_size_ = this->dictionary->add_words(int_vocabulary);
}

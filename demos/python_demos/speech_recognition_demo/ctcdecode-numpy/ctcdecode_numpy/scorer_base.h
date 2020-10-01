/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on scorer.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef SCORER_BASE_H_
#define SCORER_BASE_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "path_trie.h"
#include "word_prefix_set.h"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class ScorerBase {
public:
  ScorerBase() = delete;
  ScorerBase(double alpha,
             double beta);
  virtual ~ScorerBase();

  virtual double get_log_cond_prob(const std::vector<std::string> &words) = 0;

  double get_sent_log_prob(const std::vector<std::string> &words);

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() const { return dict_size_; }

  // return true if the language model is character based
  bool is_character_based() const { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta);

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix);

  // transform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels);

  // language model weight
  double alpha;
  // word insertion weight
  double beta;

  // pointer to the dictionary of vocabulary word prefixes
  std::unique_ptr<WordPrefixSet> dictionary;

protected:
  // necessary setup: load language model, set char map, fill word prefix dictionary
  void setup(const std::string &lm_path,
             const std::vector<std::string> &vocab_list);

  // Load language model from given path
  // This method is responsible for:
  //  * loading language model from a file
  //  * setting max_order_ and is_character_based_
  //  * setting vocabulary_
  virtual void load_lm(const std::string &lm_path) = 0;

  // fill word prefix dictionary
  void fill_dictionary(bool add_space);

  // set char map
  void set_char_map(const std::vector<std::string> &char_list);

  double get_log_prob(const std::vector<std::string> &words);

  // translate the vector in index to string
  std::string vec2str(const std::vector<int> &input);

  bool is_character_based_;
  size_t max_order_;
  std::vector<std::string> vocabulary_;

private:
  size_t dict_size_;

  int space_id_;
  std::vector<std::string> char_list_;
  std::unordered_map<std::string, int> char_map_;
};

#endif  // SCORER_BASE_H_

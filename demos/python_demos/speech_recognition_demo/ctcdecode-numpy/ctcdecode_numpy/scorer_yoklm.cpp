/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include "scorer_yoklm.h"

#include "yoklm/kenlm_v5_loader.hpp"
#include "yoklm/language_model.hpp"
#include "yoklm/vocabulary.hpp"

#include "decoder_utils.h"

ScorerYoklm::ScorerYoklm(double alpha,
                         double beta,
                         const std::string& lm_path,
                         const std::vector<std::string>& vocab_list)
      : ScorerBase(alpha, beta) {
  setup(lm_path, vocab_list);
}

ScorerYoklm::~ScorerYoklm() {}

void ScorerYoklm::load_lm(const std::string& lm_path) {
  std::unique_ptr<yoklm::KenlmV5Loader> loader(new yoklm::KenlmV5Loader);
  lm_vocabulary_.reset(new yoklm::Vocabulary);
  language_model_.reset(new yoklm::LanguageModel);
  loader->parse(lm_path);
  lm_vocabulary_->load(loader->vocabulary_config());
  language_model_->load(loader->lm_config());

  max_order_ = language_model_->order();
  vocabulary_.clear();
  vocabulary_.reserve(lm_vocabulary_->num_words());
  lm_vocabulary_->iterate_word_strings([this](yoklm::WordIndex index, std::string&& word) {
    vocabulary_.push_back(word);
  });

  is_character_based_ = true;
  for (size_t i = 0; i < vocabulary_.size(); ++i) {
    if (is_character_based_ && vocabulary_[i] != UNK_TOKEN &&
        vocabulary_[i] != START_TOKEN && vocabulary_[i] != END_TOKEN &&
        get_utf8_str_len(vocabulary_[i]) > 1) {
      is_character_based_ = false;
    }
  }
}

double ScorerYoklm::get_log_cond_prob(const std::vector<std::string>& words) {
  double cond_prob = 0;
  // avoid inserting <s> in begin
  yoklm::LmState state;
  for (size_t i = 0; i < words.size(); ++i) {
    yoklm::WordIndex word_index = lm_vocabulary_->find(words[i]);
    // encounter OOV
    if (word_index == lm_vocabulary_->unk()) {
      return OOV_SCORE;
    }
    cond_prob = language_model_->log10_p_cond(word_index, state);
  }
  // return log_e(prob)
  return cond_prob * (1./NUM_FLT_LOGE);
}

/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on scorer.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef SCORER_YOKLM_H_
#define SCORER_YOKLM_H_

#include <string>
#include <vector>
#include <memory>

#include "scorer_base.h"

namespace yoklm {
  class LanguageModel;
  class Vocabulary;
}

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion score.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class ScorerYoklm : public ScorerBase {
public:
  ScorerYoklm() = delete;
  ScorerYoklm(double alpha,
              double beta,
              const std::string &lm_path,
              const std::vector<std::string> &vocabulary);
  virtual ~ScorerYoklm();

  virtual double get_log_cond_prob(const std::vector<std::string> &words);

protected:
  // Load language model from given path
  // This method is responsible for:
  //  * loading language model from a file
  //  * setting max_order_ and is_character_based_
  //  * setting vocabulary_
  virtual void load_lm(const std::string &lm_path);

private:
  std::unique_ptr<yoklm::LanguageModel> language_model_;
  std::unique_ptr<yoklm::Vocabulary> lm_vocabulary_;
};

#endif  // SCORER_YOKLM_H_

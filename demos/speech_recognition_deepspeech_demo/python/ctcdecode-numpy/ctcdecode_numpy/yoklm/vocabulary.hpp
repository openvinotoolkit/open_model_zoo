/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#ifndef YOKLM_VOCABULARY_HPP
#define YOKLM_VOCABULARY_HPP

#include <vector>
#include <string>

#include "word_index.hpp"
#include "memory_section.hpp"


namespace yoklm {

typedef uint64_t WordHash;

struct VocabularyConfig {
  size_t num_words;  // the number of tokens in vocabulary, initialized with counts[0]
  MemorySectionArray<WordHash> word_hashes;  // sorted
  size_t num_word_strings;  // the number of tokens presented as strings. It can be 0 or num_words
  MemorySectionArray<char> word_strings;
};

// Implements MurmurHash64A
WordHash word_hash(const std::string& word);

class Vocabulary {
  public:
    Vocabulary();
    // Throw in case of an error
    void load(const VocabularyConfig& config);

    WordIndex find(WordHash word) const;
    WordIndex find(const std::string& word) const { return find(word_hash(word)); }

    // Including <unk>
    WordIndex num_words() const { return config_.num_words; }
    WordIndex unk() const { return 0; }
    WordIndex bos() const { return bos_; }
    WordIndex eos() const { return eos_; }

    bool has_word_strings() const { return config_.num_word_strings > 0; }
    // Including <unk>
    // Return false if no word strings present; true otherwise; exception in case of problems
    template <class Callback>
    bool iterate_word_strings(Callback callback) const;

  private:
    VocabularyConfig config_;
    WordIndex bos_, eos_;
};

template <class Callback>
bool Vocabulary::iterate_word_strings(Callback callback) const {
  if (config_.num_word_strings <= 0)
    return false;

  size_t word_index = 0;
  size_t prev_offset = 0;
  for (size_t offset = 0; offset < config_.word_strings.size() && word_index < config_.num_word_strings; offset++)
    if (config_.word_strings[offset] == 0) {
      callback(word_index, std::string(&config_.word_strings[prev_offset]));
      word_index++;
      prev_offset = offset + 1;
    }

  if (word_index < config_.num_word_strings)
    throw std::runtime_error("Not enough word strings in vocabulary.  Truncated LM file?");

  return true;
}

} // namespace yoklm


#endif // YOKLM_VOCABULARY_HPP

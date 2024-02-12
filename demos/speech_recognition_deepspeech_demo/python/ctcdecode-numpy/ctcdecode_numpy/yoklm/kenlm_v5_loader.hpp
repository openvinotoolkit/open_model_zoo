/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#ifndef YOKLM_KENLM_V5_LOADER_HPP
#define YOKLM_KENLM_V5_LOADER_HPP

#include <string>
#include <cstdint>
#include <stdexcept>

#include "vocabulary.hpp"
#include "language_model.hpp"
#include "memory_section.hpp"


namespace yoklm {

// It takes a provided file / file contents and a LanguageModel,
// and loads file contents into the LanguageModel.
class KenlmV5Loader {
  public:
    KenlmV5Loader();
    KenlmV5Loader(const KenlmV5Loader&) = delete;
    KenlmV5Loader& operator=(const KenlmV5Loader&) = delete;
    ~KenlmV5Loader() = default;  // ensure non-POD class for POD fields initialization

    // Throw in case of invalid format / cannot read file
    void parse(const std::string& filename);
    // Check format by checking magic number
    bool is_our_format(const MemorySection& mem);
    // Throw in case of invalid format
    void parse(MemorySection mem);
    LmConfig lm_config() const { return lm_config_; }
    VocabularyConfig vocabulary_config() const { return vocabulary_config_; }

    // Initialized to false.
    void debug_print_sections(bool print = true) const { debug_print_sections_ = print; }

  private:
    LmConfig lm_config_;
    VocabularyConfig vocabulary_config_;

    // This flag is set by _parse_header(), and is used in _parse_vocabulary_strings()
    bool with_vocabulary_strings_;

    mutable bool debug_print_sections_;

    // All _parse_* return parsed prefix length, which is guaranteed to not exceed mem section size.
    MemorySection _parse_header(MemorySection mem);
    MemorySection _parse_lm_config(MemorySection mem);
    MemorySection _parse_vocabulary(MemorySection mem);
    MemorySection _parse_lm(MemorySection mem);
    MemorySection _parse_lm_quant(MemorySection mem);
    MemorySection _parse_trie_unigram(MemorySection mem);
    MemorySection _parse_trie_medium(MemorySection mem);
    MemorySection _parse_trie_long(MemorySection mem);
    MemorySection _parse_bhiksha_highs(MemorySection mem, MediumLayer& layer_config, size_t num_entries, size_t max_value);
    MemorySection _parse_bitarray(MemorySection mem, MediumLayer& layer_config, size_t num_entries, int backoff_bits);
    MemorySection _parse_vocabulary_strings(MemorySection mem);
}; // class KenlmV5Loader

} // namespace yoklm


#endif // YOKLM_KENLM_V5_LOADER_HPP

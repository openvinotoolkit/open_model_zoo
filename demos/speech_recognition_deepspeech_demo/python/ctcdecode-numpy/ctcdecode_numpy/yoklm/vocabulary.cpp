/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <limits>
#include <cstdint>
#include <cstring>

#include "vocabulary.hpp"
#include "sorted_search.hpp"


namespace yoklm {

const uint64_t MURMUR64A_MUL = 0xc6a4a7935bd1e995ULL;
const uint64_t MURMUR64A_SHR = 47;

Vocabulary::Vocabulary() : config_(), bos_(WordIndex(-1)),
    eos_(WordIndex(-1)) {}

void Vocabulary::load(const VocabularyConfig& config) {
  config_ = config;
  bos_ = find("<s>");
  eos_ = find("</s>");
}

// Little-endian order assumed.
WordHash word_hash(const std::string& word) {
  const size_t size = word.size();

  // Assume seed = 0
  uint64_t hash = (uint64_t)word.size() * MURMUR64A_MUL;

  // C++11 guarantees countiguous storage of characters in std::string.
  // "index+7": we leave 0-byte to 7-byte chunk for later.
  for (size_t index = 0; index + 7 < size; index += 8) {
    uint64_t chunk = *(const uint64_t *)&word[index];

    chunk *= MURMUR64A_MUL;
    chunk ^= chunk >> MURMUR64A_SHR;
    chunk *= MURMUR64A_MUL;

    hash ^= chunk;
    hash *= MURMUR64A_MUL;
  }

  // Load the remaining bytes into the lower bits of chunk, padding the higher bits with 0.
  uint64_t chunk;
  if (size >= 8) {
    chunk = *(const uint64_t *)&word[size - 8];
    chunk >>= (7 - (size & 7)) * 8;
    chunk >>= 8;  // because ">> 64" is undefined behavior in C++
  } else {
    chunk = 0;
    // Struggle to avoid SEGV
    std::memcpy(&chunk, &(word[0]), size);
  }
  hash ^= chunk;
  hash *= ((size & 7) != 0) ? MURMUR64A_MUL : 1;  // an ugly exception

  hash ^= hash >> MURMUR64A_SHR;
  hash *= MURMUR64A_MUL;
  hash ^= hash >> MURMUR64A_SHR;

  return hash;
}

WordIndex Vocabulary::find(WordHash word) const {
  // WordHash == uint64_t
  WordIndex index = secant_search<MemorySectionArray<WordHash>, WordIndex, uint64_t>(
    config_.word_hashes,  // array
    0, config_.num_words-1,  // l, r;  "-1" because "<unk>" is not present in word_hashes
    0, std::numeric_limits<uint64_t>::max(),  // plv, rv
    (WordIndex)(-1),  // not_found
    word  // value
  ) + 1;  // We use overflow for not_found
  return index;
}


} // namespace yoklm

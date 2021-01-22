/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <cstring>
#include <iostream>
#include <algorithm>  // std::max

#include "kenlm_v5_loader.hpp"
#include "memory_section.hpp"


// Only support little-endian targets now.

namespace yoklm {


static const char KENLM_V5_FORMAT_MAGIC_NUMBER[0x38] = "mmap lm http://kheafield.com/code format version 5\n\0\0\0\0";  // there's an implicit \0 at the end

const int SIZE_KenlmV5SanityHeaderFormat = 0x58;
struct KenlmV5SanityHeaderFormat {
  char magic_number[0x38];
  float f32_zero, f32_one, f32_minus_half;
  uint32_t u32_one, u32_max, padding_u64_one;
  uint64_t u64_one;
};

const int SIZE_KenlmV5LmFixedParametersFormat = 0x14;
struct KenlmV5LmFixedParametersFormat {
  uint8_t lm_order;
      char padding_probing_multiplier[3];
  float probing_multiplier;  // not used with this model_type/search_type
  int32_t model_type;  // we only support 5 = quantized model with trie and arrays in Bhiksha format
  int8_t with_vocabulary_strings;  // bool
      char padding_search_type[3];
  uint32_t search_type;  // we only support 1 = trie search
};

const int SIZE_KenlmV5QuantizationHeaderFormat = 8;
struct KenlmV5QuantizationHeaderFormat {
  uint8_t quantization_type;  // we only support 2 = quantized with quantization table
  uint8_t prob_bits;
  uint8_t backoff_bits;
      uint8_t padding[5];
};

const int SIZE_KenlmV5BhikshaArrayHeaderFormat = 8;
struct KenlmV5BhikshaArrayHeaderFormat {
  uint8_t bhiksha_type;  // we only support 0, which is the only type currently
  uint8_t max_bhiksha_high_bits;
      uint8_t padding[6];
};


KenlmV5Loader::KenlmV5Loader() : lm_config_(), vocabulary_config_(),
    whole_file_(), with_vocabulary_strings_(false),
    debug_print_sections_(false) {}

void KenlmV5Loader::parse(const std::string& filename) {
  parse(load_file(filename));
}

void KenlmV5Loader::parse(MemorySection mem) {
  // Parsing requires offset alignment in some sections, so will need base address for the whole file
  whole_file_ = mem;

  mem = _parse_header(mem);
  mem = _parse_lm_config(mem);
  mem = _parse_vocabulary(mem);
  mem = _parse_lm(mem);
  mem = _parse_vocabulary_strings(mem);

  if (debug_print_sections_) {
    std::cout << "parse() end offset= " << mem.offset(whole_file_) << std::endl;
    std::cout << "parse() whole_file_ size= " << whole_file_.size() << std::endl;
  }

  whole_file_.reset();
}

bool KenlmV5Loader::is_our_format(const MemorySection& mem) {
  if (mem.size() < sizeof(KENLM_V5_FORMAT_MAGIC_NUMBER))
    return false;
  return std::memcmp(mem.ptr(), &KENLM_V5_FORMAT_MAGIC_NUMBER, sizeof(KENLM_V5_FORMAT_MAGIC_NUMBER)) == 0;
}

static void fail_sanity() {
  throw std::runtime_error("KenlmV5Loader: failed data types sanity checking in file header.");
}

MemorySection KenlmV5Loader::_parse_header(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_header offset= " << mem.offset(whole_file_) << std::endl;

  if (!is_our_format(mem))
    throw std::runtime_error("Unexpected file format. Expected kenlm v5 binary format.");
  if (sizeof(KenlmV5SanityHeaderFormat) != SIZE_KenlmV5SanityHeaderFormat)
    throw std::logic_error("Wrong size of KenlmV5SanityHeaderFormat in the code.");
  const KenlmV5SanityHeaderFormat& header = mem.at0_and_drop_prefix<KenlmV5SanityHeaderFormat>();

  if (header.f32_zero != 0.f)  fail_sanity();
  if (header.f32_one != 1.f)  fail_sanity();
  if (header.f32_minus_half != -0.5f)  fail_sanity();
  if (header.u32_one != 1)  fail_sanity();
  if (header.u32_max != (uint32_t)0xFFFFFFFF)  fail_sanity();
  if (header.u64_one != 1)  fail_sanity();

  return mem;
}

MemorySection KenlmV5Loader::_parse_lm_config(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_lm_config offset= " << mem.offset(whole_file_) << std::endl;

  if (sizeof(KenlmV5LmFixedParametersFormat) != SIZE_KenlmV5LmFixedParametersFormat)
    throw std::logic_error("Wrong size of KenlmV5LmFixedParametersFormat in the code.");
  const KenlmV5LmFixedParametersFormat& fixed_params = mem.at0_and_drop_prefix<KenlmV5LmFixedParametersFormat>();

  if (fixed_params.model_type != 5 || fixed_params.search_type != 1)
    throw std::runtime_error("KenlmV5 format: unsupported model_type/search_type. Only trie + quantization + Bhiksha arrays model type supported.");

  with_vocabulary_strings_ = (bool)fixed_params.with_vocabulary_strings;

  lm_config_.order = fixed_params.lm_order;
  if (fixed_params.lm_order < 2)
    throw std::runtime_error("Invalid KenlmV5 format: order must not be less than 2");
  if (debug_print_sections_)
    std::cout << "order= " << lm_config_.order << std::endl;

  size_t offset = 0;
  lm_config_.ngram_counts.resize(lm_config_.order);
  for (size_t k = 0; k < lm_config_.order; k++, offset += sizeof(uint64_t)) {
    lm_config_.ngram_counts[k] = mem.at<uint64_t>(offset);
    if (debug_print_sections_)
      std::cout << "count[" << k+1 << "]= " << lm_config_.ngram_counts[k] << std::endl;
  }

  // Padding to a multiple of 8, it's always 4 bytes here
  offset += 4;
  mem.drop_prefix(offset);

  return mem;
}

MemorySection KenlmV5Loader::_parse_vocabulary(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_vocabulary offset= " << mem.offset(whole_file_) << std::endl;

  const size_t num_words_to_skip = lm_config_.ngram_counts[0];  // only used in section size calculation
  // Number of words in the table of word hashes (word_hashes) + <unk>
  // "+1" for "<unk>", which in not in word_hashes table
  vocabulary_config_.num_words = mem.at0_and_drop_prefix<uint64_t>() + 1;
  if (vocabulary_config_.num_words > lm_config_.ngram_counts[0])
    throw std::runtime_error("Broken LM file: vocabulary has more words than the LM, array index overflow is possible.");

  const size_t vocabulary_size_to_skip = sizeof(uint64_t) * num_words_to_skip;
  // "-1" for "<unk>", which in not in word_hashes table
  const size_t vocabulary_size = sizeof(uint64_t) * (vocabulary_config_.num_words - 1);

  if (vocabulary_size > vocabulary_size_to_skip)
    throw std::runtime_error("Broken LM file: declared number of words in vocabulary word_hashes section is larger than section size (defined by the number of 1-grams)");
  vocabulary_config_.word_hashes = mem.get_and_drop_prefix(vocabulary_size_to_skip).prefix(vocabulary_size);

  return mem;
}

MemorySection KenlmV5Loader::_parse_lm(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_lm offset= " << mem.offset(whole_file_) << std::endl;

  if (sizeof(KenlmV5QuantizationHeaderFormat) != SIZE_KenlmV5QuantizationHeaderFormat)
    throw std::logic_error("Wrong size of KenlmV5QuantizationHeaderFormat in the code.");
  const KenlmV5QuantizationHeaderFormat& quant_header = mem.at0_and_drop_prefix<KenlmV5QuantizationHeaderFormat>();
  if (quant_header.quantization_type != 2)
    throw std::runtime_error("KenlmV5 format: unsupported quantization_type. Only quantized models are supported.");

  lm_config_.prob_bits = quant_header.prob_bits;
  lm_config_.backoff_bits = quant_header.backoff_bits;
  if (lm_config_.prob_bits < 0 || lm_config_.prob_bits > 24)
    throw std::runtime_error("KenlmV5 format: prob_bits must be from 0 to 24");
  if (lm_config_.backoff_bits < 0 || lm_config_.backoff_bits > 24)
    throw std::runtime_error("KenlmV5 format: backoff_bits must be from 0 to 24");

  mem = _parse_lm_quant(mem);
  mem = _parse_trie_unigram(mem);
  mem = _parse_trie_medium(mem);
  mem = _parse_trie_long(mem);

  return mem;
}

MemorySection KenlmV5Loader::_parse_lm_quant(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_lm_quant offset= " << mem.offset(whole_file_) << std::endl;
  if (sizeof(float) != 4)
    throw std::logic_error("KenlmV5 format: cannot work on targets with non-32-bit float type");

  lm_config_.prob_quant_tables.resize(lm_config_.order - 1);
  lm_config_.backoff_quant_tables.resize(lm_config_.order - 2);
  for (size_t order_minus_2 = 0; order_minus_2 < lm_config_.order - 2; order_minus_2++) {
    lm_config_.prob_quant_tables[order_minus_2] =    mem.get_and_drop_prefix(sizeof(float) * ((size_t)1 << lm_config_.prob_bits));
    lm_config_.backoff_quant_tables[order_minus_2] = mem.get_and_drop_prefix(sizeof(float) * ((size_t)1 << lm_config_.backoff_bits));
  }
  lm_config_.prob_quant_tables[lm_config_.order - 2] = mem.get_and_drop_prefix(sizeof(float) * ((size_t)1 << lm_config_.prob_bits));

  return mem;
}

MemorySection KenlmV5Loader::_parse_trie_unigram(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_trie_unigram offset= " << mem.offset(whole_file_) << std::endl;

  lm_config_.unigram_layer = mem.get_and_drop_prefix(sizeof(UnigramNodeFormat) * (lm_config_.ngram_counts[0] + 2));

  return mem;
}

MemorySection KenlmV5Loader::_parse_trie_medium(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_trie_medium offset= " << mem.offset(whole_file_) << std::endl;

  lm_config_.medium_layers.reserve(lm_config_.order - 1);
  lm_config_.medium_layers.resize(lm_config_.order - 2);
  for (size_t k = 2; k <= lm_config_.order - 1; k++) {  // trie layer index: k in k-gram
    mem = _parse_bhiksha_highs(mem, lm_config_.medium_layers[k-2], lm_config_.ngram_counts[k-1] + 1, lm_config_.ngram_counts[(k+1)-1]);
    mem = _parse_bitarray(mem, lm_config_.medium_layers[k-2], lm_config_.ngram_counts[k-1] + 1, lm_config_.backoff_bits);
  }

  return mem;
}

MemorySection KenlmV5Loader::_parse_trie_long(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_trie_long offset= " << mem.offset(whole_file_) << std::endl;

  lm_config_.medium_layers.resize(lm_config_.order - 1);
  MediumLayer& leaves_layer = lm_config_.medium_layers[lm_config_.order - 2];

  leaves_layer.bhiksha_total_bits = leaves_layer.bhiksha_low_bits = 0;
  leaves_layer.bhiksha_highs.reset();
  mem = _parse_bitarray(mem, leaves_layer, lm_config_.ngram_counts[lm_config_.order - 1] + 1, 0);

  return mem;
}

// Return bit length of a value. For value=0 return 0.
int required_bits(uint64_t value) {
  int bits = 0;
  while (value != 0) {
    bits++;
    value >>= 1;
  }
  return bits;
}

// Find the optimal number of "Bhiksha bits", i.e. the number of low bits
// of values to be stored in a plain array, while the high bits are to be
// stored in Bhiksha representation ("Bhiksha array").
// Expects: max_high_bits >= 0  and  value_total_bits == required_bits(max_value)
int find_bhiksha_low_bits(size_t max_index, size_t max_value, int value_total_bits, int max_high_bits) {
  // WHY ON THE EARTH it's not just stored in the file?

  if (max_high_bits < 0)
    throw std::logic_error("Internal error: max_high_bits < 0");

  int min_allowed_low_bits = std::max(value_total_bits - max_high_bits, 0);
  int best_low_bits = -1;
  int64_t best_size = 0;  // best means lowest size (and highest low_bits among the lowest size)
  // There's always at least one iteration of the loop, as max_high_bits>=0
  for (int low_bits = min_allowed_low_bits; low_bits <= value_total_bits; low_bits++) {
    int64_t cur_size = (max_value >> low_bits) * 64 - max_index * (int64_t)(value_total_bits - low_bits);
    if (best_low_bits < 0 || best_size >= cur_size) {
      best_size = cur_size;
      best_low_bits = low_bits;
    }
  }
  return best_low_bits;
}

MemorySection KenlmV5Loader::_parse_bhiksha_highs(MemorySection mem, MediumLayer& layer_config, size_t num_entries, size_t max_value) {
  if (debug_print_sections_)
    std::cout << "_parse_bhiksha_highs offset= " << mem.offset(whole_file_) << std::endl;

  if (sizeof(KenlmV5BhikshaArrayHeaderFormat) != SIZE_KenlmV5BhikshaArrayHeaderFormat)
    throw std::logic_error("Wrong size of KenlmV5BhikshaArrayHeaderFormat in the code.");
  const KenlmV5BhikshaArrayHeaderFormat& bhiksha_header = mem.at0_and_drop_prefix<KenlmV5BhikshaArrayHeaderFormat>();
  if (bhiksha_header.bhiksha_type != 0)
    throw std::runtime_error("KenlmV5 format: unsupported bhiksha_type.  Probably, it was created by a too new version of kenlm.");
  // No need to check bhiksha_header.max_bhiksha_high_bits>=0, as it's uint8_t

  const int total_bits = layer_config.bhiksha_total_bits = required_bits(max_value);
  const int low_bits = layer_config.bhiksha_low_bits =
    find_bhiksha_low_bits(num_entries, max_value, total_bits, bhiksha_header.max_bhiksha_high_bits);
  layer_config.bhiksha_highs_count = (max_value >> low_bits) + 1;
  const size_t bhiksha_highs_size = sizeof(uint64_t) * layer_config.bhiksha_highs_count;

  MemorySection unaligned_bhiksha_highs = mem.get_and_drop_prefix(bhiksha_highs_size + 7);  // Add 7 bytes for alignment

  // Align offset to a multiple of 8
  size_t alignment = (-unaligned_bhiksha_highs.offset(whole_file_)) & 7;
  layer_config.bhiksha_highs = unaligned_bhiksha_highs.subsection(alignment, bhiksha_highs_size);
  if (debug_print_sections_)
    std::cout << "_parse_bhiksha_highs aligned offset= " << layer_config.bhiksha_highs.offset(whole_file_) << std::endl;

  if (layer_config.bhiksha_highs_count == 0 || layer_config.bhiksha_highs[0] != 0)
    std::runtime_error("Broken LM file: bhisha_highs[0] != 0");

  return mem;
}

uint64_t make_bitmask(int bits) { return ((uint64_t)1 << bits) - 1; }

MemorySection KenlmV5Loader::_parse_bitarray(MemorySection mem, MediumLayer& layer_config, size_t num_entries, int backoff_bits) {
  if (debug_print_sections_)
    std::cout << "_parse_bitarray offset= " << mem.offset(whole_file_) << std::endl;

  const int word_index_bits = required_bits(lm_config_.ngram_counts[0]);
  const int bits_per_record = word_index_bits + lm_config_.prob_bits + backoff_bits + layer_config.bhiksha_low_bits;
  const size_t bitarray_size = (num_entries * bits_per_record + 7) / 8 + 8;
  layer_config.backoff_bits = backoff_bits;

  // Initialize bit field descriptors
  int offset = 0;
  layer_config.word_field.offset = offset;
  layer_config.word_field.mask = make_bitmask(word_index_bits);
  offset += word_index_bits;

  layer_config.backoff_field.offset = offset;
  layer_config.backoff_field.mask = make_bitmask(backoff_bits);
  offset += backoff_bits;

  layer_config.prob_field.offset = offset;
  layer_config.prob_field.mask = make_bitmask(lm_config_.prob_bits);
  offset += lm_config_.prob_bits;

  layer_config.bhiksha_low_field.offset = offset;
  layer_config.bhiksha_low_field.mask = make_bitmask(layer_config.bhiksha_low_bits);
  offset += layer_config.bhiksha_low_bits;

  // Parse bitarray
  layer_config.bit_array = mem.get_and_drop_prefix(bitarray_size);
  layer_config.bit_array.set_stride(bits_per_record);
  layer_config.bit_array.set_bit_field(layer_config.word_field);
  if (debug_print_sections_)
    std::cout << "_parse_bitarray num_entries= " << num_entries << std::endl;

  return mem;
}

MemorySection KenlmV5Loader::_parse_vocabulary_strings(MemorySection mem) {
  if (debug_print_sections_)
    std::cout << "_parse_vocabulary_strings offset= " << mem.offset(whole_file_) << std::endl;

  if (!with_vocabulary_strings_) {
    vocabulary_config_.num_word_strings = 0;
    vocabulary_config_.word_strings.reset();
    return mem;
  }

  if (std::memcmp(mem.prefix(6).ptr(), "<unk>", 6) != 0)
    throw std::runtime_error("Wrong KenlmV5 format: vocabulary strings must start with \"<unk>\" token.  Broken LM file?");

  size_t words_left = vocabulary_config_.num_word_strings = vocabulary_config_.num_words;
  size_t offset;
  for (offset = 0; offset < mem.size() && words_left > 0; offset++)
    words_left -= (int)(mem[offset] == 0);

  if (words_left > 0)
    throw std::runtime_error("Not enough word strings in vocabulary.  Truncated LM file?");

  vocabulary_config_.word_strings = mem.get_and_drop_prefix(offset);

  return mem;
}

} // namespace yoklm

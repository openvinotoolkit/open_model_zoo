/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>

#include "memory_section.hpp"


namespace yoklm {

MemorySection::MemorySection() : ptr_(nullptr), base_ptr_(nullptr), size_(0), owner_() {
  const uint64_t endianness = 0x0102030405060708ULL;
  if (((const uint8_t *)&endianness)[0] != 8)
    throw std::logic_error("yoklm requires little-endian byte order.");
}

MemorySection::MemorySection(const MemorySection& ms)
    : ptr_(ms.ptr_), base_ptr_(ms.base_ptr_), size_(ms.size_), owner_(ms.owner_) {}

MemorySection& MemorySection::operator=(const MemorySection& ms) {
  if (&ms != this) {
    size_ = 0;
    ptr_ = ms.ptr_;
    base_ptr_ = ms.base_ptr_;
    size_ = ms.size_;
    owner_ = ms.owner_;
  }
  return *this;
}

MemorySection::MemorySection(std::shared_ptr<ManagedMemory> mm)
    : ptr_(mm->ptr()), base_ptr_(mm->ptr()), size_(mm->size()), owner_(mm) {}

MemorySection::MemorySection(uint8_t const * ptr, uint8_t const * base_ptr, size_t size, std::shared_ptr<ManagedMemory>& owner)
    : ptr_(ptr), base_ptr_(base_ptr), size_(size), owner_(owner) {}

MemorySection MemorySection::subsection(size_t start, size_t size) const {
  if (size > size_ || start > size_ - size)
    throw std::out_of_range("Requested subsection exceeds MemorySection boundary");
  return MemorySection(ptr_ + start, base_ptr_, size, owner_);
}

MemorySection MemorySection::without_prefix(size_t skip) const {
  if (skip > size_) {
    throw std::out_of_range(
      "Not enough data to skip the requested number of bytes: requested " +
      std::to_string((unsigned long)skip) +
      ", section size " +
      std::to_string((unsigned long)size_)
    );
  }
  return MemorySection(ptr_ + skip, base_ptr_, size_ - skip, owner_);
}

MemorySection MemorySection::prefix(size_t size) const {
  if (size > size_) {
    throw std::out_of_range(
      "Not enough data to fetch the requested number of bytes: requested " +
      std::to_string((unsigned long)size) +
      ", section size " +
      std::to_string((unsigned long)size_)
    );
  }
  return MemorySection(ptr_, base_ptr_, size, owner_);
}

void MemorySection::drop_prefix(size_t skip) {
  if (skip > size_) {
    throw std::out_of_range(
      "Not enough data to skip the requested number of bytes: requested " +
      std::to_string((unsigned long)skip) +
      ", section size " +
      std::to_string((unsigned long)size_)
    );
  }
  size_ -= skip;
  ptr_ += skip;
}

MemorySection MemorySection::get_and_drop_prefix(size_t size) {
  if (size > size_) {
    throw std::out_of_range(
      "Not enough data to skip the requested number of bytes: requested " +
      std::to_string((unsigned long)size) +
      ", section size " +
      std::to_string((unsigned long)size_)
    );
  }
  const uint8_t * ptr_orig = ptr_;
  size_ -= size;
  ptr_ += size;
  return MemorySection(ptr_orig, base_ptr_, size, owner_);
}

void MemorySection::reset() {
  size_ = 0;
  ptr_ = base_ptr_ = nullptr;
  owner_.reset();
}

void MemorySectionBitArray::set_stride(int stride) {
  stride_ = stride;
  // Dividing max(size_t) by 2 to be sure there're no overflow in operator(),
  // because BitField::offset being an int cannot exceed max(size_t)/2 (when it's non-negative).
  // Plus in all actual cases in yoklm library, BitField::offset is not more than ~200.
  if (stride <= 0 || size() / size_t(stride) >= std::numeric_limits<size_t>::max() / 8 / 2)
    throw std::range_error("Array size exceeds size_t in MemorySectionBitArray. Broken LM file?");
  if (size() >= 8)
    // Subtracting 8 to ensure no access beyond our memory section to exclude SIGSEGV on read.
    // Kenlm binary format has an 8-byte padding after each bit array for this.
    index_limit_ = (size() - 8) * 8 / stride;
  else
    index_limit_ = 0;
}

MemorySection load_file(const std::string& filename) {
  std::ifstream is;
  is.open(filename, std::ios::in | std::ios::binary);
  if (!is)
    throw std::runtime_error("Cannot open file: " + filename);

  is.seekg(0, std::ios::end);
  if (uintmax_t(is.tellg()) >= std::numeric_limits<size_t>::max())
    throw std::range_error("File size exceeds size_t: " + filename);
  size_t file_length = is.tellg();
  is.seekg(0, std::ios::beg);

  std::shared_ptr<ManagedMemory> mm(std::make_shared<ManagedMemory>(file_length));

  is.read(reinterpret_cast<char *>(mm->ptr()), file_length);
  if (size_t(is.gcount()) != file_length)
    throw std::runtime_error("Some problem while reading file: " + filename);

  return MemorySection(mm);
}


} // namespace yoklm

/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>

#include "memory_section.hpp"


namespace yoklm {

MemorySection::MemorySection() : ptr_(nullptr), size_(0), owner_() {}
MemorySection::MemorySection(const MemorySection& ms) : ptr_(ms.ptr_), size_(ms.size_), owner_(ms.owner_) {}
MemorySection& MemorySection::operator=(const MemorySection& ms) {
  if (&ms != this) {
    size_ = 0;
    ptr_ = ms.ptr_;
    size_ = ms.size_;
    owner_ = ms.owner_;
  }
  return *this;
}

MemorySection::MemorySection(std::shared_ptr<ManagedMemory> mm) : ptr_(mm->ptr()), size_(mm->size()), owner_(mm) {}
MemorySection::MemorySection(size_t size) : MemorySection(std::make_shared<ManagedMemory>(size)) {}
MemorySection::MemorySection(uint8_t const * ptr, size_t size, std::shared_ptr<ManagedMemory>& owner)
    : ptr_(ptr), size_(size), owner_(owner) {}

MemorySection MemorySection::subsection(size_t start, size_t size) const {
  if (size > size_ || start > size_ - size)
    throw std::out_of_range("Requested subsection exceeds MemorySection boundary");
  return MemorySection(ptr_ + start, size, owner_);
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
  return MemorySection(ptr_ + skip, size_ - skip, owner_);
}

MemorySection MemorySection::prefix(size_t size) const {
  if (size > size_) {
    throw std::out_of_range(
      "Not enough data to get the requested number of bytes: requested " +
      std::to_string((unsigned long)size) +
      ", section size " +
      std::to_string((unsigned long)size_)
    );
  }
  return MemorySection(ptr_, size, owner_);
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
  return MemorySection(ptr_orig, size, owner_);
}

void MemorySection::reset() {
  size_ = 0;
  ptr_ = nullptr;
  owner_.reset();
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

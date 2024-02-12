/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#ifndef YOKLM_MEMORY_SECTION_HPP
#define YOKLM_MEMORY_SECTION_HPP

#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>


#include "memory_section.hpp"


namespace yoklm {

// This class provides a virtual interface to manage a section of memory.
class ManagedMemory {
  public:
    ManagedMemory() = delete;
    ManagedMemory(const ManagedMemory& mm) = delete;
    ManagedMemory& operator=(const ManagedMemory& mm) = delete;
    virtual ~ManagedMemory() { delete[] ptr_; }

    ManagedMemory(size_t size) : ptr_(new uint8_t[size]), size_(size) {}
  private:
    // The caller transfers ownership of *ptr.
    // ptr must have been created with new uint8_t[...] in case of the default destructor.
    ManagedMemory(uint8_t * ptr, size_t size) : ptr_(ptr), size_(size) {}
  public:

    virtual uint8_t * ptr() const { return ptr_; }
    virtual size_t size() const { return size_; }

  private:
    uint8_t * ptr_;
    size_t size_;
}; // class ManagedMemory

// Holds a non-owning pointer+size to a CONSTANT memory section, that
// is memory contents is not going to be changed using this pointer.
// Plus it holds a shared_ptr to ManagedMemory object that owns the memory.
class MemorySection {
  public:
    MemorySection();
    MemorySection(const MemorySection& ms);
    MemorySection& operator=(const MemorySection& ms);

    MemorySection(std::shared_ptr<ManagedMemory> mm);
  private:
    // Initialize with unmanaged memory. The caller is responsible for proper *ptr lifetime.
    // base_ptr can point anywhere and is never dereferenced, used only in base_offset() method.
    MemorySection(const uint8_t * ptr, const uint8_t * base_ptr, size_t size, std::shared_ptr<ManagedMemory>& owner);

  public:
    const uint8_t& operator[](size_t index) const {
      if (index >= size_)
        throw std::out_of_range("Out of bounds access in MemorySection. Broken LM file?");
      return ptr_[index];
    }
    template <typename T> const T& at(size_t index) const {
      if (size_ < sizeof(T) || index > size_ - sizeof(T))
        throw std::out_of_range("Out of bounds access in MemorySection::at<T>(). Broken LM file?");
      return *reinterpret_cast<const T *>(&ptr_[index]);
    }
    // Return subsection
    MemorySection subsection(size_t start, size_t size) const;
    // Return a prefix subsection section
    MemorySection prefix(size_t size) const;
    // Return section with prefix removed
    MemorySection without_prefix(size_t skip) const;
    // Remove prefix in-place
    void drop_prefix(size_t skip);
    // Remove prefix in-place and return the prefix
    MemorySection get_and_drop_prefix(size_t size);
    // Remove prefix in-place and return the prefix, prefix size and type = T.
    // Beware of T& lifetime!
    template<typename T> const T& at0_and_drop_prefix() {
      return get_and_drop_prefix(sizeof(T)).at<T>(0);
    }

    void reset();
    const uint8_t * ptr() const { return ptr_; }
    size_t size() const { return size_; }

    void set_base_offset(std::ptrdiff_t base_offset) { base_ptr_ = ptr_ - base_offset; }
    std::ptrdiff_t base_offset() const { return ptr_ - base_ptr_; }

  private:
    const uint8_t * ptr_;
    const uint8_t * base_ptr_;  // this pointer is never dereferenced, only for expressions like (ptr - base_ptr_)
    size_t size_;
    mutable std::shared_ptr<ManagedMemory> owner_;
}; // class MemorySection

// Essentially this class template only overloads operator[] in MemorySection.
template <typename T>
class MemorySectionArray : public MemorySection {
  public:
    MemorySectionArray() {}
    MemorySectionArray(const MemorySection& ms) : MemorySection(ms) {}

    // Here :index: is not offset, but index into T[] array instead. That is, the second element's index is 1.
    const T& operator[](size_t index) const {
      if (index >= size() / sizeof(T))
        throw std::out_of_range(
          "Out of bounds access in MemorySectionArray: index="
            + std::to_string((unsigned long)index)
            + " size=" + std::to_string((unsigned long)(size() / sizeof(T)))
            + ". Broken LM file?"
        );
      return reinterpret_cast<const T *>(ptr())[index];
    }
}; // class MemorySectionArray

struct BitField {
  int offset;  // in bits
  uint64_t mask;
}; // struct BitField

// Essentially this class template only overloads operator[] in MemorySection.
class MemorySectionBitArray : public MemorySection {
  public:
    MemorySectionBitArray() : stride_(0), bit_field_{}, index_limit_(0) {}
    MemorySectionBitArray(const MemorySection& ms)  // ms must contain 8-byte padding after the actual bit array
        : MemorySection(ms), stride_(0), bit_field_{}, index_limit_(0) {}

    // Defined in header file for efficiency.
    // Expects (0 <= bf.offset) and (bf.offset + bits(bf.mask) <= stride).
    uint64_t operator()(size_t index, const BitField& bf) const {
      size_t bit_index = index * stride_ + bf.offset;
      if (index >= index_limit_)
        throw std::logic_error("Out of bounds access in MemorySectionBitArray. Broken LM file?");
      uint64_t data = *reinterpret_cast<const uint64_t *>(&ptr()[bit_index / 8]) >> (bit_index & 7);  // unaligned read
      return data & bf.mask;
    }
    uint64_t operator[](size_t index) const { return operator()(index, bit_field_); }

    void set_stride(int stride);
    void set_bit_field(const BitField& bf) { bit_field_ = bf; }
    int stride() const { return stride_; }

  private:
    int stride_;  // for operator[] and operator()
    BitField bit_field_;  // for operator[]
    size_t index_limit_;
}; // class MemorySectionBitArray

// Throws an exception if cannot.
MemorySection load_file(const std::string& filename);

} // namespace yoklm


#endif // YOKLM_MEMORY_SECTION_HPP

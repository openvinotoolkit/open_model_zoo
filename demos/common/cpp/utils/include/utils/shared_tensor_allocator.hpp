/*
// Copyright (C) 2021-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include <opencv2/core.hpp>
#include <openvino/runtime/allocator.hpp>

// To prevent false-positive clang compiler warning
// (https://github.com/openvinotoolkit/openvino/pull/11092#issuecomment-1073846256):
// warning: destructor called on non-final 'SharedTensorAllocator' that has virtual functions
// but non-virtual destructor [-Wdelete-non-abstract-non-virtual-dtor]
// SharedTensorAllocator class declared as final

class SharedTensorAllocator final : public ov::AllocatorImpl {
public:
    SharedTensorAllocator(const cv::Mat& img) : img(img) {}

    ~SharedTensorAllocator() = default;

    void* allocate(const size_t bytes, const size_t) override {
        return bytes <= img.rows * img.step[0] ? img.data : nullptr;
    }

    void deallocate(void* handle, const size_t bytes, const size_t) override {}

    bool is_equal(const AllocatorImpl& other) const override {
        auto other_tensor_allocator = dynamic_cast<const SharedTensorAllocator*>(&other);
        return other_tensor_allocator != nullptr && other_tensor_allocator == this;
    }

private:
    const cv::Mat img;
};

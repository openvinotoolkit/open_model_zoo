/*
// Copyright (C) 2018-2020 Intel Corporation
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
#include "ie_allocator.hpp"
#include "opencv2/core.hpp"

class SharedBlobAllocator : public InferenceEngine::IAllocator {
public:
    SharedBlobAllocator(const cv::Mat& img);
    ~SharedBlobAllocator();
    virtual void* lock(void* handle, InferenceEngine::LockOp op = InferenceEngine::LOCK_FOR_WRITE) noexcept;
    virtual void unlock(void* handle) noexcept;
    virtual void* alloc(size_t size) noexcept;
    virtual bool free(void* handle) noexcept;

private:
    const cv::Mat img;
};

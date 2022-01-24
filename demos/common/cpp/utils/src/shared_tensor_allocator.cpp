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

#include "utils/shared_tensor_allocator.hpp"

SharedBlobAllocator::SharedBlobAllocator(const cv::Mat & img) :
    img(img) {
}

SharedBlobAllocator::~SharedBlobAllocator() {
}

void * SharedBlobAllocator::lock(void * handle, InferenceEngine::LockOp op) noexcept {
    if(handle == img.data) {
        return img.data;
    }
    return nullptr;
}

void SharedBlobAllocator::unlock(void * handle) noexcept {
}

void * SharedBlobAllocator::alloc(size_t size) noexcept {
    return size <= img.rows*img.step[0] ? img.data : nullptr;
}

bool SharedBlobAllocator::free(void * handle) noexcept {
    return false;
}

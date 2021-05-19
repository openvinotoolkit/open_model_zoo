/*
// Copyright (C) 2018-2021 Intel Corporation
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

#include <unordered_map>
#include <future>
#include "fourcc.h"

class VaApiContext;
class VaApiImage;

class VaSurfacesPool {
  public:
    VASurfaceID acquire(uint16_t width, uint16_t height, FourCC fourcc);
    void release(const VaApiImage& img);
    void waitForCompletion();
    VaSurfacesPool() : display(nullptr) {}
    VaSurfacesPool(VADisplay display) : display(display) {}
    ~VaSurfacesPool();
  private:
    using Element = std::pair<VASurfaceID, bool>; // second is true if image is in use
    std::unordered_multimap<uint64_t, Element> images;
    std::condition_variable _free_image_condition_variable;
    std::mutex mtx;

    uint64_t calcKey(uint16_t width, uint16_t height, FourCC fourcc) {
        return static_cast<uint64_t>(fourcc) |
            ((static_cast<uint64_t>(width) & 0xFFFF)<<32) | ((static_cast<uint64_t>(height) & 0xFFFF)<<48);
    }

    VADisplay display;
};

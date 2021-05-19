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

#include "vaapi_image_pool.h"
#include "vaapi_images.h"
#include "vaapi_context.h"

VaSurfacesPool::~VaSurfacesPool() {
    waitForCompletion();
}

VASurfaceID VaSurfacesPool::acquire(uint16_t width, uint16_t height, FourCC fourcc) {
    uint64_t key = calcKey(width, height, fourcc);
    std::unique_lock<std::mutex> lock(mtx);

    auto its = images.equal_range(key);

    for (auto it = its.first; it != its.second; ++it) {
        auto& foundElement = it->second;
        if (!foundElement.second) {
            foundElement.second = true;
            return foundElement.first;
        }
    }

    auto& elem = images.emplace(key,(Element(
        VaApiContext::createSurface(display,width,height,fourcc),
        true)))->second;

    return elem.first;
}

void VaSurfacesPool::release(const VaApiImage& img) {
    std::unique_lock<std::mutex> lock(mtx);
    auto its = images.equal_range(calcKey(img.width, img.height, img.format));

    for (auto it = its.first; it != its.second; ++it) {
        auto& foundElement = it->second;
        if(foundElement.first == img.va_surface_id) {
            foundElement.second = false;
            _free_image_condition_variable.notify_one();
            return;
        }
    }
    throw std::runtime_error("VaSurfacesPool: An attempt to release non-pooled surface is detected");
}

void VaSurfacesPool::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mtx);
    for (auto& imagePair : images) {
        while(imagePair.second.second)
            _free_image_condition_variable.wait(lock);
    }
}

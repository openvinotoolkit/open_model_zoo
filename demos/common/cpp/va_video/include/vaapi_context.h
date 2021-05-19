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

#include "vaapi_utils.h"

#include <functional>
#include <stdexcept>

#include <va/va.h>
#include <gpu/gpu_context_api_va.hpp>

#include <cstdint>
#include <type_traits>
#include <memory>
#include <vaapi_image_pool.h>

class VaApiContext
{
  private:
    VADisplay vaDisplay = nullptr;
    VAConfigID vaConfig = VA_INVALID_ID;
    VAContextID vaContextId = VA_INVALID_ID;
    int driFileDescriptor = 0;
    bool isOwningVaDisplay = false;
    InferenceEngine::gpu::VAContext::Ptr gpuSharedContext = nullptr;

  public:
    using Ptr=std::shared_ptr<VaApiContext>;
    VaApiContext(VADisplay display = nullptr);
    VaApiContext(VADisplay display, InferenceEngine::Core& coreForSharedContext);
    VaApiContext(InferenceEngine::Core& coreForSharedContext);

    ~VaApiContext();

    void createSharedContext(InferenceEngine::Core& core);

    VAContextID contextId() {
      return vaContextId;
    }

    VADisplay display() {
      return vaDisplay;
    }

    InferenceEngine::gpu::VAContext::Ptr sharedContext() {
      return gpuSharedContext;
    }

    VaSurfacesPool& getSurfacesPool(){ return *surfacesPool;}

    static VASurfaceID createSurface(VADisplay display, uint16_t width, uint16_t height, FourCC format);

  private:
    void create(VADisplay display);
    void close();
    std::unique_ptr<VaSurfacesPool> surfacesPool;

    static void messageErrorCallback(void *, const char *message);
    static void messageInfoCallback(void *, const char *message);
};

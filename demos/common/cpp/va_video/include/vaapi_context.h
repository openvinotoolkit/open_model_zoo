/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

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

namespace InferenceBackend
{

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

} // namespace InferenceBackend

/*******************************************************************************
 * Copyright (C) 2018-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include <va/va.h>
#include "vaapi_context.h"

namespace InferenceBackend {

class VaApiImageMap_SytemMemory : public ImageMap {
  public:
    VaApiImageMap_SytemMemory();
    ~VaApiImageMap_SytemMemory();

    Image Map(const Image &image) override;
    void Unmap() override;

  protected:
    VADisplay va_display;
    VAImage va_image;
};

class VaApiImageMap_VASurafce : public ImageMap {
  public:
    VaApiImageMap_VASurafce();
    ~VaApiImageMap_VASurafce();

    Image Map(const Image &image) override;
    void Unmap() override;
};

} // namespace InferenceBackend

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

#include <cstdint>
#include <type_traits>
#include <memory>

namespace
{

template <int a, int b, int c, int d>
struct fourcc
{
    enum { code = (a) | (b << 8) | (c << 16) | (d << 24) };
};

} // namespace

namespace InferenceBackend
{

enum FourCC {
    FOURCC_NONE = 0,
    FOURCC_RGBP_F32 = 0x07282024,
    FOURCC_NV12 = fourcc<'N', 'V', '1', '2'>::code,
    FOURCC_BGRA = fourcc<'B', 'G', 'R', 'A'>::code,
    FOURCC_BGRX = fourcc<'B', 'G', 'R', 'X'>::code,
    FOURCC_BGRP = fourcc<'B', 'G', 'R', 'P'>::code,
    FOURCC_BGR = fourcc<'B', 'G', 'R', ' '>::code,
    FOURCC_RGBA = fourcc<'R', 'G', 'B', 'A'>::code,
    FOURCC_RGBX = fourcc<'R', 'G', 'B', 'X'>::code,
    FOURCC_RGB = fourcc<'R', 'G', 'B', ' '>::code,
    FOURCC_RGBP = fourcc<'R', 'G', 'B', 'P'>::code,
    FOURCC_I420 = fourcc<'I', '4', '2', '0'>::code
};

class VaApiContext
{
  private:
    VADisplay vaDisplay = nullptr;
    VAConfigID vaConfig = VA_INVALID_ID;
    VAContextID vaContextId = VA_INVALID_ID;
    int driFileDescriptor = 0;
    bool isOwningVaDisplay = false;

  public:
    using Ptr=std::shared_ptr<VaApiContext>;
    explicit VaApiContext(VADisplay display);
    VaApiContext();

    ~VaApiContext();

    VAContextID contextId() {
      return vaContextId;
    }

    VADisplay display() {
      return vaDisplay;
    }
};

} // namespace InferenceBackend

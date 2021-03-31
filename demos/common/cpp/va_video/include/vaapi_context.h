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
    VADisplay _va_display = nullptr;
    VAConfigID _va_config = VA_INVALID_ID;
    VAContextID _va_context_id = VA_INVALID_ID;
    int _dri_file_descriptor = 0;
    bool _own_va_display = false;
    std::function<void(const char *)> message_callback;

  public:
    explicit VaApiContext(VADisplay va_display);
    VaApiContext();

    ~VaApiContext();

    VADisplay Display();
    VAContextID Id();
};

} // namespace InferenceBackend

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

enum class MemoryType {
    ANY = 0,
    SYSTEM = 1,
    DMA_BUFFER = 2,
    VAAPI = 3,
};

enum FourCC {
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

template <typename T>
struct Rectangle
{
    static_assert(std::is_floating_point<T>::value or std::is_integral<T>::value,
                  "struct Rectangle can only be instantiated with numeric type types");

    T x;
    T y;
    T width;
    T height;
};

struct Image
{
    MemoryType type;
    static const uint32_t MAX_PLANES_NUMBER = 4;
    union {
        uint8_t *planes[MAX_PLANES_NUMBER]; // if type==SYSTEM
        int dma_fd;                         // if type==DMA_BUFFER
        struct {                            // if type==VAAPI
            uint32_t va_surface_id;
            void *va_display;
        };
    };
    int format; // FourCC
    uint32_t width;
    uint32_t height;
    uint32_t size;
    uint32_t stride[MAX_PLANES_NUMBER];
    uint32_t offsets[MAX_PLANES_NUMBER];
    Rectangle<uint32_t> rect;
};

// Map DMA/VAAPI image into system memory
class ImageMap
{
  public:
    virtual Image Map(const Image &image) = 0;
    virtual void Unmap() = 0;

    static ImageMap *Create(MemoryType type);
    virtual ~ImageMap() = default;
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

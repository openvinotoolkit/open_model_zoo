/*******************************************************************************
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <future>
#include <memory>
#include <vector>

#include "vaapi_context.h"

#include <opencv2/core.hpp>

namespace InferenceBackend {
class VaApiImagePool;

class VaApiImage{
  friend class VaApiImagePool;
  public:
    VaApiImage(){};
    VaApiImage(VADisplay va_display, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface = VA_INVALID_ID);
    VaApiImage(VaApiImage&& other);
    VaApiImage&& operator=(VaApiImage&& other);

    enum CONVERSION_TYPE {
      CONVERT_TO_RGB,
      CONVERT_TO_BGR,
      CONVERT_COPY
    };

    cv::Mat CopyToMat(CONVERSION_TYPE convType = CONVERT_TO_BGR);

    uint32_t va_surface_id = VA_INVALID_ID;
    VADisplay va_display = nullptr;

    FourCC format = FOURCC_NONE; // FourCC
    uint32_t width = 0 ;
    uint32_t height = 0;

  protected:
    std::atomic_bool completed;
 
    VaApiImage(const VaApiImage&other) = delete;
    void DestroyImage();

    VASurfaceID CreateVASurface();
    static int FourCCToVART(FourCC fourcc);
};

class VaPooledImage{
  public:
    using Ptr = std::shared_ptr<VaPooledImage>;
    VaPooledImage(VaApiImage* img, VaApiImagePool* pool);
    ~VaPooledImage();

    VaApiImage* image = nullptr;
  protected:
    VaApiImagePool* pool = nullptr;
};

class VaApiImagePool {
  public:
    VaPooledImage::Ptr Acquire();
    void Release(VaPooledImage::Ptr& image);
    void Release(VaApiImage* image);
    struct ImageInfo {
        size_t width;
        size_t height;
        FourCC format;
    };
    void Flush();
    VaApiImagePool(VaApiContext *context_, size_t image_pool_size, ImageInfo info);
    ~VaApiImagePool();
  private:
    std::vector<std::unique_ptr<VaApiImage>> _images;
    std::condition_variable _free_image_condition_variable;
    std::mutex _free_images_mutex;

    VaApiContext *context;
    ImageInfo info;
};

} // namespace InferenceBackend

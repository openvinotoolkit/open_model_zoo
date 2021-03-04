/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <future>
#include <memory>
#include <vector>

#include "vaapi_context.h"
#include "vaapi_image_map.h"

namespace InferenceBackend {
class VaApiImagePool;

class VaApiImage : public Image{
  friend class VaApiImagePool;
  public:
    VaApiContext *context = nullptr;

  protected:
    std::atomic_bool completed;

    VaApiImage(const VaApiImage &other) = delete;
    VaApiImage(VaApiContext *context_, uint32_t width, uint32_t height, FourCC format, MemoryType memory_type);
    void DestroyImage();
};

class VaPooledImage{
  public:
    using Ptr = std::shared_ptr<VaPooledImage>;
    VaPooledImage(VaApiImage* img, VaApiImagePool* pool);
    ~VaPooledImage();

    VaApiImage* image;
  protected:
    VaApiImagePool* pool=nullptr;

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
        MemoryType memory_type;
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

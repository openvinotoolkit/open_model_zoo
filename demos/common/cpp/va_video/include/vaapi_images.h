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
#include <va/va.h>
#include <va/va_drmcommon.h>

#include <opencv2/core.hpp>

namespace InferenceBackend {
class VaApiImagePool;

class VaApiImage{
  friend class VaApiImagePool;
  public:
    enum RESIZE_MODE {
      RESIZE_FILL,
      RESIZE_KEEP_ASPECT,
      RESIZE_KEEP_ASPECT_LETTERBOX
    };
    
    enum CONVERSION_TYPE {
      CONVERT_TO_RGB,
      CONVERT_TO_BGR,
      CONVERT_COPY
    };

  public:
    VaApiImage() {};
    VaApiImage(const VaApiContext::Ptr& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface = VA_INVALID_ID, bool autoDestroySurface=true);
    virtual ~VaApiImage() { if(autoDestroySurface) DestroyImage(); }    

    using Ptr = std::shared_ptr<VaApiImage>;

    VaApiImage::Ptr CloneToAnotherContext(const VaApiContext::Ptr& newContext);
    void ResizeTo(VaApiImage::Ptr dstImage, VaApiImage::RESIZE_MODE resizeMode = RESIZE_FILL);

    cv::Mat CopyToMat(CONVERSION_TYPE convType = CONVERT_TO_BGR);

    uint32_t va_surface_id = VA_INVALID_ID;
    VaApiContext::Ptr context = nullptr;

    FourCC format = FOURCC_NONE; // FourCC
    uint32_t width = 0 ;
    uint32_t height = 0;

  protected:
    std::atomic_bool completed;
    bool autoDestroySurface;
 
    VaApiImage(const VaApiImage& other) = delete;
    void DestroyImage();

    VASurfaceID CreateVASurface();
    static int FourCCToVART(FourCC fourcc);
};

class VaPooledImage : public VaApiImage{
  public:
    using Ptr = std::shared_ptr<VaPooledImage>;
    VaPooledImage(const VaApiContext::Ptr& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface, VaApiImagePool* pool) :
      VaApiImage(context, width, height, format, va_surface, false), pool(pool) {}
    ~VaPooledImage() override;

  protected:
    VaApiImagePool* pool = nullptr;
};

class VaApiImagePool {
  public:
    VaApiImage::Ptr Acquire();
    void Release(VaApiImage* image);
    struct ImageInfo {
        size_t width;
        size_t height;
        FourCC format;
    };
    void WaitForCompletion();
    VaApiImagePool(const VaApiContext::Ptr& context_, size_t image_pool_size, ImageInfo info);
    ~VaApiImagePool();
  private:
    using Element = std::pair<std::unique_ptr<VaApiImage>, bool>;
    std::vector<std::pair<std::unique_ptr<VaApiImage>, bool>> images; // second is true if image is in use
    std::condition_variable _free_image_condition_variable;
    std::mutex mtx;

    VaApiContext::Ptr context;
    ImageInfo info;
};

} // namespace InferenceBackend

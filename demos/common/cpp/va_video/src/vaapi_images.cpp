/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "vaapi_images.h"
#include <iostream>

using namespace InferenceBackend;

namespace {

VASurfaceID CreateVASurface(VADisplay dpy, uint32_t width, uint32_t height, FourCC format,
                            int rt_format = VA_RT_FORMAT_YUV420) {
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = format;

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(dpy, VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1));
    if (not(format_attrib.value & rt_format))
        throw std::invalid_argument("Unsupported runtime format for surface.");

    VASurfaceID va_surface_id;
    VA_CALL(vaCreateSurfaces(dpy, rt_format, width, height, &va_surface_id, 1, &surface_attrib, 1))
    return va_surface_id;
}

} // namespace

VaApiImage::VaApiImage(VaApiContext *context_, uint32_t width, uint32_t height, FourCC format, MemoryType memory_type) :
    completed(true),
    context(context_) {
    this->type = memory_type;
    this->width = width;
    this->height = height;
    this->format = format;
    this->va_display = context->Display();
    this->va_surface_id = CreateVASurface(va_display, width, height, format);
}

void VaApiImage::DestroyImage() {
    if (type == MemoryType::VAAPI && va_surface_id && va_surface_id != VA_INVALID_ID) {
        try {
            VA_CALL(vaDestroySurfaces(va_display, (uint32_t *)&va_surface_id, 1));
        } catch (const std::exception &e) {
            std::string error_message = std::string("VA surface destroying failed with exception: ") + e.what();
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

VaPooledImage::VaPooledImage(VaApiImage* img, VaApiImagePool* pool) :
    pool(pool),
    image(img) {
};

VaPooledImage::~VaPooledImage() {
    if(pool)
        pool->Release(image);
}


VaApiImagePool::VaApiImagePool(VaApiContext *context_, size_t image_pool_size, ImageInfo info) :
    context(context_),
    info(info)
{
    if (not context_)
        throw std::invalid_argument("VaApiContext is nullptr");
    for (size_t i = 0; i < image_pool_size; ++i) {
        _images.push_back(std::unique_ptr<VaApiImage>(
            new VaApiImage(context_, info.width, info.height, info.format, info.memory_type)));
    }
}

VaApiImagePool::~VaApiImagePool() {
}

VaPooledImage::Ptr VaApiImagePool::Acquire() {
    std::unique_lock<std::mutex> lock(_free_images_mutex);
    for (;;) {
        for (auto &image : _images) {
            if (image->completed) {
                image->completed = false;
                return VaPooledImage::Ptr(new VaPooledImage(image.get(),this));
            }
        }
        _images.push_back(std::unique_ptr<VaApiImage>(
            new VaApiImage(context, info.width, info.height, info.format, info.memory_type)));
        _images.back()->completed=false;
        return VaPooledImage::Ptr(new VaPooledImage(_images.back().get(),this));
    }
}

void VaApiImagePool::Release(VaPooledImage::Ptr& imagePtr) {
    Release(imagePtr->image);
}

void VaApiImagePool::Release(VaApiImage* image) {
    if (!image)
        throw std::runtime_error("Received VA-API image is null");

    std::unique_lock<std::mutex> lock(_free_images_mutex);
    image->completed = true;
    _free_image_condition_variable.notify_one();
}

void VaApiImagePool::Flush() {
    std::unique_lock<std::mutex> lock(_free_images_mutex);
    for (auto &image : _images) {
        std::unique_lock<std::mutex> lock(_free_images_mutex);
        while(!image->completed)
            _free_image_condition_variable.wait(lock);
    }
}

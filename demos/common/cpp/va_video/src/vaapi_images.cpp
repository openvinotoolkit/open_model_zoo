/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "vaapi_images.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace InferenceBackend {
    int VaApiImage::FourCCToVART(FourCC fourcc)
    {
        switch(fourcc)
        {
            case FOURCC_NV12:
            case FOURCC_I420:
                return VA_RT_FORMAT_YUV420;
            case FOURCC_BGRA:
            case FOURCC_BGRX:
            case FOURCC_BGR:    
            case FOURCC_RGBA:
            case FOURCC_RGBX:
            case FOURCC_RGB:
                return VA_RT_FORMAT_RGB32;
            case FOURCC_BGRP:
            case FOURCC_RGBP:
                return VA_RT_FORMAT_RGBP;
            default:
                break;
        }
        throw std::invalid_argument("Cannot convert FOURCC to RT_FORMAT.");
    }

VASurfaceID VaApiImage::CreateVASurface() {
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = format;

    int rt_format = FourCCToVART(format);

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(va_display, VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1));
    if (!(format_attrib.value & rt_format))
        throw std::invalid_argument("Unsupported runtime format for surface.");

    VASurfaceID va_surface_id;
    VA_CALL(vaCreateSurfaces(va_display, rt_format, width, height, &va_surface_id, 1, &surface_attrib, 1))
    return va_surface_id;
}

VaApiImage::VaApiImage(VADisplay va_display, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface) {
    completed = true;
    this->width = width;
    this->height = height;
    this->format = format;
    this->va_display = va_display;
    this->va_surface_id = va_surface == VA_INVALID_ID ? CreateVASurface() : va_surface;
}

VaApiImage::VaApiImage( VaApiImage&& other)
{
    *this = std::move(other);
}

VaApiImage&& VaApiImage::operator=(VaApiImage&& other)
{
    this->completed = (bool)other.completed;
    this->width = other.width;
    this->height = other.height;
    this->format = other.format;
    this->va_display = other.va_display;
    this->va_surface_id = other.va_surface_id;

    other.completed = true;
    other.width = 0;
    other.height = 0;
    other.format = FOURCC_NONE;
    other.va_display = nullptr;
    other.va_surface_id = VA_INVALID_ID;
}


void VaApiImage::DestroyImage() {
    if (va_surface_id != VA_INVALID_ID) {
        try {
            VA_CALL(vaDestroySurfaces(va_display, (uint32_t *)&va_surface_id, 1));
        } catch (const std::exception &e) {
            std::string error_message = std::string("VA surface destroying failed with exception: ") + e.what();
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

cv::Mat VaApiImage::CopyToMat(VaApiImage::CONVERSION_TYPE convType) {

    VAImage mappedImage;
    void *pSurface = nullptr;
    cv::Mat outMat;
    cv::Mat mappedMat;

    //--- Mapping image
    VA_CALL(vaDeriveImage(va_display, va_surface_id, &mappedImage))
    VA_CALL(vaMapBuffer(va_display, mappedImage.buf, &pSurface))

    //--- Copying data to Mat. Only NV12/I420 foramts are supported now
    switch(format) {
        case FOURCC_NV12:
        case FOURCC_I420:
            mappedMat = cv::Mat(mappedImage.height*3/2,mappedImage.width,CV_8UC1,pSurface,{mappedImage.pitches[0]});
            break;
        default:
            throw std::invalid_argument("VAApiImage Map: non-supported FOURCC encountered");
    }

    //--- Converting image
    switch(convType) {
        case CONVERT_TO_RGB:
            cv::cvtColor(mappedMat,outMat,format == FOURCC_NV12 ? cv::COLOR_YUV2RGB_NV12 : cv::COLOR_YUV2RGB);
            break;
        case CONVERT_TO_BGR:
            cv::cvtColor(mappedMat,outMat,format == FOURCC_NV12 ? cv::COLOR_YUV2BGR_NV12 : cv::COLOR_YUV2BGR);
            break;
        default:
            mappedMat.copyTo(outMat);
            break;
    }

    //--- Unmapping image
    try {
        VA_CALL(vaUnmapBuffer(va_display, mappedImage.buf))
        VA_CALL(vaDestroyImage(va_display, mappedImage.image_id))
        va_display=0;
    } catch (const std::exception &e) {
        std::string error_message =
            std::string("VA buffer unmapping (destroying) failed with exception: ") + e.what();
        std::cout << error_message.c_str() << std::endl;
    }

    return outMat;
}


VaPooledImage::VaPooledImage(VaApiImage* img, VaApiImagePool* pool) :
    image(img),
    pool(pool) {
};

VaPooledImage::~VaPooledImage() {
    if(pool)
        pool->Release(image);
}


VaApiImagePool::VaApiImagePool(VaApiContext *context_, size_t image_pool_size, ImageInfo info) :
    context(context_),
    info(info)
{
    if (!context_)
        throw std::invalid_argument("VaApiContext is nullptr");
    for (size_t i = 0; i < image_pool_size; ++i) {
        _images.push_back(std::unique_ptr<VaApiImage>(
            new VaApiImage(context_->Display(), info.width, info.height, info.format)));
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
            new VaApiImage(context, info.width, info.height, info.format)));
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
}
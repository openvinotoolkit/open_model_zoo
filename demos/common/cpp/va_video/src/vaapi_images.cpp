/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "vaapi_images.h"
#include <iostream>
#include <unistd.h>
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

VaApiImage::Ptr VaApiImage::CloneToAnotherContext(const VaApiContext::Ptr& newContext)
{
    int rtFormat = FourCCToVART(format);

    VADRMPRIMESurfaceDescriptor drm_descriptor = VADRMPRIMESurfaceDescriptor();
    VA_CALL(vaExportSurfaceHandle(context->display(), va_surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                                  VA_EXPORT_SURFACE_READ_ONLY, &drm_descriptor));

    VASurfaceAttribExternalBuffers external = VASurfaceAttribExternalBuffers();
    external.width = drm_descriptor.width;
    external.height = drm_descriptor.height;
    external.pixel_format = drm_descriptor.fourcc;

    if (drm_descriptor.num_objects != 1)
        throw std::invalid_argument("Unexpected objects number");
    auto object = drm_descriptor.objects[0];
    external.num_buffers = 1;
    uint64_t dma_fd = object.fd;
    external.buffers = &dma_fd;
    external.data_size = object.size;

    external.num_planes = drm_descriptor.num_layers;
    for (uint32_t i = 0; i < external.num_planes; i++) {
        auto layer = drm_descriptor.layers[i];
        if (layer.num_planes != 1)
            throw std::invalid_argument("Unexpected planes number");
        external.pitches[i] = layer.pitch[0];
        external.offsets[i] = layer.offset[0];
    }

    VASurfaceAttrib attribs[2] = {};
    attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attribs[0].type = VASurfaceAttribMemoryType;
    attribs[0].value.type = VAGenericValueTypeInteger;
    attribs[0].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME;

    attribs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attribs[1].type = VASurfaceAttribExternalBufferDescriptor;
    attribs[1].value.type = VAGenericValueTypePointer;
    attribs[1].value.value.p = &external;

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(newContext->display(), VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1));
    if (!(format_attrib.value & rtFormat))
        throw std::invalid_argument("Unsupported runtime format for surface.");

    VASurfaceID surfaceID = VA_INVALID_SURFACE;
    VA_CALL(vaCreateSurfaces(newContext->display(), rtFormat, drm_descriptor.width, drm_descriptor.height, &surfaceID, 1,
                             attribs, 2));
//!!!                             std::cout<<"SurfID "<<surfaceID<<std::endl;
    return VaApiImage::Ptr(new VaApiImage(newContext,external.width,external.height,format,surfaceID),
        [dma_fd](VaApiImage* img) {
            if (close(dma_fd) == -1) {
                throw std::runtime_error("VaApiConverter::Convert: close fd failed");
            }
            delete img;
        }
    );
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
    VA_CALL(vaGetConfigAttributes(context->display(), VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1));
    if (!(format_attrib.value & rt_format))
        throw std::invalid_argument("Unsupported runtime format for surface.");

    VASurfaceID va_surface_id;
    VA_CALL(vaCreateSurfaces(context->display(), rt_format, width, height, &va_surface_id, 1, &surface_attrib, 1))
                                 std::cout<<"Created "<<va_surface_id<<std::endl;
    return va_surface_id;
}

VaApiImage::VaApiImage(const VaApiContext::Ptr& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface, bool autoDestroySurface) {
    this->completed = true;
    this->width = width;
    this->height = height;
    this->format = format;
    this->context = context;
    this->autoDestroySurface = autoDestroySurface;
//!!!    if(va_surface!=VA_INVALID_ID) {
//        std::cout<<"Received: "<<va_surface<<std::endl;
//    }
    this->va_surface_id = va_surface == VA_INVALID_ID ? CreateVASurface() : va_surface;
}


void VaApiImage::DestroyImage() {
    if (va_surface_id != VA_INVALID_ID) {
        try {
//!!!            std::cout<<"Destr:"<<va_surface_id<<std::endl;
            VA_CALL(vaDestroySurfaces(context->display(), (uint32_t *)&va_surface_id, 1));
        } catch (const std::exception &e) {
            std::string error_message = std::string("VA surface destroying failed with exception: ") + e.what();
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

cv::Mat VaApiImage::CopyToMat(VaApiImage::CONVERSION_TYPE convType) {

    VAImage mappedImage;
    void *pData = nullptr;
    cv::Mat outMat;
    cv::Mat mappedMat;

    //--- Mapping image
    VA_CALL(vaDeriveImage(context->display(), va_surface_id, &mappedImage))
    VA_CALL(vaMapBuffer(context->display(), mappedImage.buf, &pData))

    //--- Copying data to Mat. Only NV12/I420 formats are supported now
    switch(format) {
        case FOURCC_NV12:
        case FOURCC_I420:
            mappedMat = cv::Mat(mappedImage.height*3/2,mappedImage.width,CV_8UC1,pData,{mappedImage.pitches[0]});
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

    try {
        VA_CALL(vaUnmapBuffer(context->display(), mappedImage.buf))
        VA_CALL(vaDestroyImage(context->display(), mappedImage.image_id));
    } catch (const std::exception &e) {
        std::string error_message =
            std::string("VA buffer unmapping (destroying) failed with exception: ") + e.what();
        std::cout << error_message.c_str() << std::endl;
    }

    return outMat;
}

void VaApiImage::ResizeTo(VaApiImage::Ptr dstImage, VaApiImage::RESIZE_MODE resizeMode) {
    if(context->display() != dstImage->context->display() || context->contextId() != dstImage->context->contextId())
    {
        throw std::invalid_argument("ResizeTo: (context, display) of the source and destination images should be the same");
    }

    VAProcPipelineParameterBuffer pipelineParam = VAProcPipelineParameterBuffer();
    pipelineParam.surface = va_surface_id;
    VARectangle surface_region = {.x = 0,
                                  .y = 0,
                                  .width = (uint16_t)this->width,
                                  .height = (uint16_t)this->height};
    if (surface_region.width > 0 && surface_region.height > 0)
        pipelineParam.surface_region = &surface_region;

    //pipelineParam.filter_flags = VA_FILTER_SCALING_HQ; // High-quality scaling method

    VABufferID pipelineParamBufId = VA_INVALID_ID;
    VA_CALL(vaCreateBuffer(context->display(), context->contextId(), VAProcPipelineParameterBufferType,
                           sizeof(pipelineParam), 1, &pipelineParam, &pipelineParamBufId));

    VA_CALL(vaBeginPicture(context->display(), context->contextId(), dstImage->va_surface_id))

    VA_CALL(vaRenderPicture(context->display(), context->contextId(), &pipelineParamBufId, 1))

    VA_CALL(vaEndPicture(context->display(), context->contextId()))

    VA_CALL(vaDestroyBuffer(context->display(), pipelineParamBufId))
}

VaPooledImage::~VaPooledImage() {
    pool->Release(this);
}

VaApiImagePool::VaApiImagePool(const VaApiContext::Ptr& context_, size_t image_pool_size, ImageInfo info) :
    context(context_),
    info(info)
{
    if (!context_)
        throw std::invalid_argument("VaApiContext is nullptr");
    for (size_t i = 0; i < image_pool_size; ++i) {
        images.push_back(Element(std::unique_ptr<VaApiImage>(new VaApiImage(context, info.width, info.height, info.format)), false));
    }
}

VaApiImagePool::~VaApiImagePool() {
    WaitForCompletion();
}

VaApiImage::Ptr VaApiImagePool::Acquire() {
    std::unique_lock<std::mutex> lock(mtx);
    for (;;) {
        for (auto &imagePair : images) {
            if (!imagePair.second) {
                imagePair.second = true;
                auto& image= imagePair.first;
                return std::make_shared<VaPooledImage>(image->context,image->width, image->height, image->format, image->va_surface_id, this);
            }
        }

        images.push_back(Element(std::unique_ptr<VaApiImage>(new VaApiImage(context, info.width, info.height, info.format)),true));
        auto& image = images.back().first;
        return std::make_shared<VaPooledImage>(image->context,image->width, image->height, image->format, image->va_surface_id, this);
    }
}

void VaApiImagePool::Release(VaApiImage* image) {
    if (!image)
        throw std::runtime_error("VaApiImagePool: Received image ptr is null");

    std::unique_lock<std::mutex> lock(mtx);
    for(auto& imagePair : images) {
        if(imagePair.first->va_surface_id == image->va_surface_id &&
            imagePair.first->context->display() == image->context->display()) {
            imagePair.second = false;
            _free_image_condition_variable.notify_one();
            return;
        }
    }
    throw std::runtime_error("VaApiImagePool: An attempt to release non-pooled image is deteted");
}

void VaApiImagePool::WaitForCompletion() {
    std::unique_lock<std::mutex> lock(mtx);
    for (auto &imagePair : images) {
        while(imagePair.second)
            _free_image_condition_variable.wait(lock);
    }
}
}
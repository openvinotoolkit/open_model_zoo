/*******************************************************************************
 * Copyright (C) 2019-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "vaapi_images.h"
#include <iostream>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <vaapi_context.h>

namespace InferenceBackend {

VaApiImage::Ptr VaApiImage::cloneToAnotherContext(const VaApiContext::Ptr& newContext)
{
    int rtFormat = fourCCToVART(format);

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

    return VaApiImage::Ptr(new VaApiImage(newContext,external.width,external.height,format,surfaceID),
        [dma_fd](VaApiImage* img) {
            if (close(dma_fd) == -1) {
                throw std::runtime_error("VaApiConverter::Convert: close fd failed");
            }
            delete img;
        }
    );
}


VASurfaceID VaApiImage::createVASurface() {
    return VaApiContext::createSurface(context->display(),width,height,format);
}

VaApiImage::VaApiImage(const VaApiContext::Ptr& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface, bool autoDestroySurface) {
    this->width = width;
    this->height = height;
    this->format = format;
    this->context = context;
    this->autoDestroySurface = autoDestroySurface;
    this->va_surface_id = va_surface == VA_INVALID_ID ? createVASurface() : va_surface;
}


void VaApiImage::destroyImage() {
    if (va_surface_id != VA_INVALID_ID) {
        try {
            VA_CALL(vaDestroySurfaces(context->display(), (uint32_t *)&va_surface_id, 1));
        } catch (const std::exception &e) {
            std::string error_message = std::string("VA surface destroying failed with exception: ") + e.what();
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

cv::Mat VaApiImage::copyToMat(IMG_CONVERSION_TYPE convType) {

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

VaApiImage::Ptr VaApiImage::resizeUsingPooledSurface(uint16_t width, uint16_t height, IMG_RESIZE_MODE resizeMode, bool hqResize) {
    auto surfID = context->getSurfacesPool().acquire(width,height,format);
    auto img = std::make_shared<VaPooledImage>(context,width,height,format,surfID);
    resizeTo(img,resizeMode,hqResize);
    return img;
}

void VaApiImage::resizeTo(VaApiImage::Ptr dstImage, IMG_RESIZE_MODE resizeMode, bool hqResize) {
    if(context->display() != dstImage->context->display() || context->contextId() != dstImage->context->contextId())
    {
        throw std::invalid_argument("resizeTo: (context, display) of the source and destination images should be the same");
    }

    VAProcPipelineParameterBuffer pipelineParam = VAProcPipelineParameterBuffer();
    pipelineParam.surface = va_surface_id;
    VARectangle surface_region = {.x = 0,
                                  .y = 0,
                                  .width = (uint16_t)this->width,
                                  .height = (uint16_t)this->height};
    if (surface_region.width > 0 && surface_region.height > 0)
        pipelineParam.surface_region = &surface_region;

    pipelineParam.filter_flags = hqResize ? VA_FILTER_SCALING_HQ : VA_FILTER_SCALING_DEFAULT;

    VABufferID pipelineParamBufId = VA_INVALID_ID;
    VA_CALL(vaCreateBuffer(context->display(), context->contextId(), VAProcPipelineParameterBufferType,
                           sizeof(pipelineParam), 1, &pipelineParam, &pipelineParamBufId));

    VA_CALL(vaBeginPicture(context->display(), context->contextId(), dstImage->va_surface_id))

    VA_CALL(vaRenderPicture(context->display(), context->contextId(), &pipelineParamBufId, 1))

    VA_CALL(vaEndPicture(context->display(), context->contextId()))

    VA_CALL(vaDestroyBuffer(context->display(), pipelineParamBufId))
}

VaPooledImage::~VaPooledImage() {
    context->getSurfacesPool().release(*this);
}

}

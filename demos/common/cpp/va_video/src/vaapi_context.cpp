#include "vaapi_context.h"

#include <iostream>
#include <cassert>
#include <tuple>

#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>

using namespace InferenceBackend;

int fourCCToVART(FourCC fourcc)
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

void VaApiContext::messageErrorCallback(void *, const char *message)
{
    std::cerr << message << std::endl;
}

void VaApiContext::messageInfoCallback(void *, const char *message)
{
    std::cerr << message << std::endl;
}

VaApiContext::VaApiContext(VADisplay display) {
    create(display);
}

VaApiContext::VaApiContext(VADisplay display, InferenceEngine::Core& coreForSharedContext) {
    create(display);
    createSharedContext(coreForSharedContext);
}

VaApiContext::VaApiContext(InferenceEngine::Core& coreForSharedContext) {
    create(nullptr);
    createSharedContext(coreForSharedContext);
}

void VaApiContext::create(VADisplay display) {
    if (!display) {
        driFileDescriptor = open("/dev/dri/renderD128", O_RDWR);
        if (!driFileDescriptor) {
            throw std::runtime_error("Error opening /dev/dri/renderD128");
        }

        vaDisplay = vaGetDisplayDRM(driFileDescriptor);
        if (!vaDisplay) {
            close();
            throw std::runtime_error("Error opening VAAPI Display");
        }

        vaSetErrorCallback(vaDisplay, messageErrorCallback, nullptr);
        vaSetInfoCallback(vaDisplay, messageInfoCallback, nullptr);
        int major_version = 0, minor_version = 0;
        VA_CALL(vaInitialize(vaDisplay, &major_version, &minor_version));

        isOwningVaDisplay = true;
    }
    else {
        vaDisplay=display;
        isOwningVaDisplay = false;
    }

    VAConfigAttrib attrib;
    attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(vaDisplay, VAProfileNone, VAEntrypointVideoProc, &attrib, 1));

    vaConfig = 0;
    VA_CALL(vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, &attrib, 1, &vaConfig));
    if (vaConfig == 0) {
        close();
        throw std::invalid_argument("Could not create VA config. Cannot initialize VaApiContext without VA config.");
    }

    vaContextId = 0;
    VA_CALL(vaCreateContext(vaDisplay, vaConfig, 0, 0, VA_PROGRESSIVE, nullptr, 0, &vaContextId));
    if (vaContextId == 0) {
        close();
        throw std::invalid_argument("Could not create VA context. Cannot initialize VaApiContext without VA context.");
    }

    surfacesPool = std::unique_ptr<VaSurfacesPool>(new VaSurfacesPool(vaDisplay));
}

VaApiContext::~VaApiContext() {
    close();
}

void VaApiContext::close() {
    if (vaContextId != VA_INVALID_ID) {
        vaDestroyContext(vaDisplay, vaContextId);
    }
    if (vaConfig != VA_INVALID_ID) {
        vaDestroyConfig(vaDisplay, vaConfig);
    }
    if (vaDisplay && isOwningVaDisplay) {
        VAStatus status = vaTerminate(vaDisplay);
        if (status != VA_STATUS_SUCCESS) {
            std::string error_message =
                std::string("VA Display termination failed with code ") + std::to_string(status);
            std::cerr << error_message.c_str() << std::endl;
        }
        int status_code = ::close(driFileDescriptor);
        if (status_code != 0) {
            std::string error_message =
                std::string("DRI file descriptor closing failed with code ") + std::to_string(status_code);
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

void VaApiContext::createSharedContext(InferenceEngine::Core& core)
{
    gpuSharedContext = InferenceEngine::gpu::make_shared_context(core, "GPU", display());
}

VASurfaceID VaApiContext::createSurface(VADisplay display, uint16_t width, uint16_t height, FourCC format) {
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = format;

    int rt_format = fourCCToVART(format);

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(display, VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1));
    if (!(format_attrib.value & rt_format))
        throw std::invalid_argument("Unsupported runtime format for surface.");

    VASurfaceID va_surface_id;
    VA_CALL(vaCreateSurfaces(display, rt_format, width, height, &va_surface_id, 1, &surface_attrib, 1))
    return va_surface_id;
}
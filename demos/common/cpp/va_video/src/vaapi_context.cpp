#include "vaapi_context.h"

#include <iostream>
#include <cassert>
#include <tuple>

#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>

using namespace InferenceBackend;

namespace {

static void message_callback_error(void *user_context, const char *message)
{
    (void)user_context;
    std::cerr << message << std::endl;
}

static void message_callback_info(void *user_context, const char *message)
{
    (void)user_context;
    std::cerr << message << std::endl;
}

std::tuple<VADisplay, int> create_va_display_and_device_descriptor()
{
    int dri_file_descriptor = open("/dev/dri/renderD128", O_RDWR);
    if (!dri_file_descriptor) {
        throw std::runtime_error("Error opening /dev/dri/renderD128");
    }

    VADisplay display = vaGetDisplayDRM(dri_file_descriptor);
    if (!display) {
        throw std::runtime_error("Error opening VAAPI Display");
    }

    vaSetErrorCallback(display, message_callback_error, nullptr);
    vaSetInfoCallback(display, message_callback_info, nullptr);
    int major_version = 0, minor_version = 0;
    VA_CALL(vaInitialize(display, &major_version, &minor_version));

    return std::make_tuple(display, dri_file_descriptor);
}

std::tuple<VAConfigID, VAContextID> create_config_and_context(VADisplay display,
                                                              int surface_rt_format = VA_RT_FORMAT_YUV420) {
    if (!display) {
        throw std::invalid_argument("VADisplay is nullptr. Cannot initialize VaApiContext without VADisplay.");
    }
    VAConfigAttrib attrib;
    attrib.type = VAConfigAttribRTFormat;
    VA_CALL(vaGetConfigAttributes(display, VAProfileNone, VAEntrypointVideoProc, &attrib, 1));
    if (not(attrib.value & surface_rt_format))
        throw std::invalid_argument("Unsupported runtime format for surface.");
    VAConfigID config_id = 0;
    VA_CALL(vaCreateConfig(display, VAProfileNone, VAEntrypointVideoProc, &attrib, 1, &config_id));
    if (config_id == 0) {
        throw std::invalid_argument("Could not create VA config. Cannot initialize VaApiContext without VA config.");
    }
    VAContextID context_id = 0;
    VA_CALL(vaCreateContext(display, config_id, 0, 0, VA_PROGRESSIVE, nullptr, 0, &context_id));
    if (context_id == 0) {
        throw std::invalid_argument("Could not create VA context. Cannot initialize VaApiContext without VA context.");
    }
    return std::make_tuple(config_id, context_id);
}

} // namespace

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
        std::tie(vaDisplay, driFileDescriptor) = create_va_display_and_device_descriptor();
        isOwningVaDisplay = true;
    }
    else {
        vaDisplay=display;
        isOwningVaDisplay = false;
    }

    std::tie(vaConfig, vaContextId) = create_config_and_context(vaDisplay);
}

VaApiContext::~VaApiContext() {
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
        int status_code = close(driFileDescriptor);
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

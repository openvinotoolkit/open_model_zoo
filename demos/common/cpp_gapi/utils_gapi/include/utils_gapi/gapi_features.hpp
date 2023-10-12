#pragma once

#include <opencv2/core/version.hpp>
#include <opencv2/gapi/core.hpp>

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 2)
    #define GAPI_IE_BACKEND
#endif

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR > 5)
    #define GAPI_ONNX_BACKEND
#endif

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 8)
    #define GAPI_OV_BACKEND
#endif

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR > 8)
    #define GAPI_ONNX_BACKEND_EP_EXTENSION
#endif

#pragma once

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR > 8
    #define GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
#endif


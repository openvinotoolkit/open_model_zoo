// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utils/args_helper.hpp>

#include "utils_gapi/onnx_backend.hpp"

#ifdef GAPI_ONNX_BACKEND
const std::initializer_list<std::string> getONNXSupportedEP() {
    static const std::initializer_list<std::string> providers{
        "CPU"
#ifdef GAPI_ONNX_BACKEND_EP_EXTENSION
        , "DML", "OV"
#endif
    };
    return providers;
}
#endif // GAPI_ONNX_BACKEND

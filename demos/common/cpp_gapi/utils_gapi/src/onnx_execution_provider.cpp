// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_gapi/onnx_execution_provider.hpp"


const std::initializer_list<std::string> getONNXSupportedEP() {
    static const std::initializer_list<std::string> providers{
        "CPU"
#ifdef GAPI_ONNX_BACKEND_EP_EXTENSION
        , "DML", "OVEP"
#endif
    };
    return providers;
}

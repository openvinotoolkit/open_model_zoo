// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_gapi/kernel_package.hpp"

namespace util {
cv::gapi::GKernelPackage getKernelPackage(const std::string& type) {
    if (type == "opencv") {
        return cv::gapi::combine(cv::gapi::core::cpu::kernels(), cv::gapi::imgproc::cpu::kernels());
    } else if (type == "fluid") {
        return cv::gapi::combine(cv::gapi::core::fluid::kernels(), cv::gapi::imgproc::fluid::kernels());
    } else {
        throw std::logic_error("Unsupported kernel package type: " + type);
    }
    GAPI_Assert(false && "Unreachable code!");
}
} // namespace util

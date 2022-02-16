// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/infer.hpp>

namespace nets {
using GMat2 = std::tuple<cv::GMat,cv::GMat>;
G_API_NET(TextDetection,           <GMat2(cv::GMat)>,                   "textDetect");
G_API_NET(TextRecognition,         <cv::GMat(cv::GMat)>,                "textRecogn");
G_API_NET(TextRecognitionEncoding, <GMat2(cv::GMat)>,                   "textRecognEncoding");
G_API_NET(TextRecognitionDecoding, <GMat2(cv::GMat,cv::GMat,cv::GMat)>, "textRecognDecoding");
} // namespace nets

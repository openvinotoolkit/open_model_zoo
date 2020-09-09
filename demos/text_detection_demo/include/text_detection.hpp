// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

std::vector<cv::RotatedRect> postProcess(const InferenceEngine::BlobMap &blobs, const cv::Size& image_size,
                                         const cv::Size& image_shape, float cls_conf_threshold,
                                         float link_conf_threshold);

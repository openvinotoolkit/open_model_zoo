// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>


struct IndexScore {
    using IndicesStorage = std::list<size_t>;
    using ConfidenceMap = std::multimap<float, typename IndicesStorage::iterator>;

    using ClassDescription = std::tuple<float, size_t, std::string>;
    using LabelsStorage = std::vector<ClassDescription>;

    IndexScore() = default;

    void getScoredLabels(const std::vector<std::string> &in_labes,
                         LabelsStorage &out_scored_labels_to_append) const;
    static IndexScore create_from_array(const float *out_blob_data_ptr, size_t out_blob_element_count,
                                        size_t top_k_amount);
private:
    ConfidenceMap max_confidence_with_indices;
    IndicesStorage max_element_indices;
    size_t classification_indices_scale;
};

using GRect = cv::GOpaque<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;
using GIndexScore = cv::GOpaque<IndexScore>;

namespace custom {
G_API_OP(CentralCrop, <GRect(GSize)>, "classification_benchmark.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(TopK, <GIndexScore(cv::GMat, uint32_t)>, "classification_benchmark.custom.post_processing") {
    static cv::GOpaqueDesc outMeta(const cv::GMatDesc &, uint32_t) {
        return cv::empty_gopaque_desc();
    }
};

cv::gapi::GKernelPackage kernels();

}  // namespace custom

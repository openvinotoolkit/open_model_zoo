// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "face_reid.hpp"
#include "tracker.hpp"

namespace {
float ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2) {
    float xy = static_cast<float>(descr1.dot(descr2));
    float xx = static_cast<float>(descr1.dot(descr1));
    float yy = static_cast<float>(descr2.dot(descr2));
    float norm = sqrt(xx * yy) + 1e-6f;
    return 1.0f - xy / norm;
}

}  // namespace

const char EmbeddingsGallery::unknown_label[] = "Unknown";
const int EmbeddingsGallery::unknown_id = TrackedObject::UNKNOWN_LABEL_IDX;

std::vector<int> EmbeddingsGallery::GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const {
    if (embeddings.empty() || idx_to_id.empty())
        return std::vector<int>(embeddings.size(), unknown_id);
    cv::Mat distances(static_cast<int>(embeddings.size()), static_cast<int>(idx_to_id.size()), CV_32F);
    for (int i = 0; i < distances.rows; i++) {
        int k = 0;
        for (size_t j = 0; j < identities.size(); j++) {
            for (const auto& reference_emb : identities[j].embeddings) {
                distances.at<float>(i, k) = ComputeReidDistance(embeddings[i], reference_emb);
                k++;
            }
        }
    }
    KuhnMunkres matcher(use_greedy_matcher);
    auto matched_idx = matcher.Solve(distances);
    std::vector<int> output_ids;
    for (auto col_idx : matched_idx) {
        if (distances.at<float>(output_ids.size(), col_idx) > reid_threshold)
            output_ids.push_back(unknown_id);
        else
            output_ids.push_back(idx_to_id[col_idx]);
    }
    return output_ids;
}

std::string EmbeddingsGallery::GetLabelByID(int id) const {
    if (id >= 0 && id < static_cast<int>(identities.size()))
        return identities[id].label;
    else
        return unknown_label;
}

size_t EmbeddingsGallery::size() const {
    return identities.size();
}

std::vector<std::string> EmbeddingsGallery::GetIDToLabelMap() const {
    std::vector<std::string> map;
    map.reserve(identities.size());
    for (const auto& item : identities) {
        map.emplace_back(item.label);
    }
    return map;
}

bool EmbeddingsGallery::LabelExists(const std::string& label) const {
    return identities.end() != std::find_if(identities.begin(), identities.end(), [label](const GalleryObject& o) {
               return o.label == label;
           });
}

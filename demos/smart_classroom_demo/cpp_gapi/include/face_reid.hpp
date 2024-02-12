// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "detector.hpp"

enum class RegistrationStatus {
    SUCCESS,
    FAILURE_LOW_QUALITY,
    FAILURE_NOT_DETECTED,
};

struct GalleryObject {
    std::vector<cv::Mat> embeddings;
    std::string label;
    int id;

    GalleryObject(const std::vector<cv::Mat>& embeddings, const std::string& label, int id)
        : embeddings(embeddings),
          label(label),
          id(id) {}
};

class EmbeddingsGallery {
public:
    static const char unknown_label[];
    static const int unknown_id;
    EmbeddingsGallery(double threshold,
                      const std::vector<GalleryObject>& identities_m,
                      const std::vector<int>& idx_to_id_m,
                      bool use_greedy_matcher = false)
        : reid_threshold(threshold),
          use_greedy_matcher(use_greedy_matcher),
          identities(identities_m),
          idx_to_id(idx_to_id_m) {}
    size_t size() const;
    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string GetLabelByID(int id) const;
    std::vector<std::string> GetIDToLabelMap() const;
    bool LabelExists(const std::string& label) const;

private:
    double reid_threshold;
    bool use_greedy_matcher;
    std::vector<GalleryObject> identities;
    std::vector<int> idx_to_id;
};

void AlignFaces(std::vector<cv::Mat>* face_images, std::vector<cv::Mat>* landmarks_vec);
